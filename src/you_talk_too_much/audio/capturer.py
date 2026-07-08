import math
import queue
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)


class AudioCapturer:
    """Audio Capturer using sounddevice with a single-threaded tick pattern."""

    TARGET_RATE = 16000
    MIN_SAMPLES = TARGET_RATE * 5  # 5 seconds minimum before VAD check
    VAD_TAIL_SAMPLES = 24000  # 1.5 seconds for silence detection

    def __init__(self, on_audio_ready: Callable[[np.ndarray], None]) -> None:
        """Initialize the audio capturer."""
        logger.info("Initializing Audio Capturer...")

        self.on_audio_ready = on_audio_ready
        self._chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._vad_model = load_silero_vad()
        self._resample_down: int = 1
        self._resample_up: int = 1

    def start(self) -> None:
        """Start capturing audio by opening the sounddevice stream."""
        logger.info("Starting audio capture...")

        self._buffer.clear()
        _drain_all(self._chunk_queue)

        native_rate = int(sd.query_devices(kind="input")["default_samplerate"])
        gcd = math.gcd(self.TARGET_RATE, native_rate)
        self._resample_up = self.TARGET_RATE // gcd
        self._resample_down = native_rate // gcd

        logger.info(
            "Device native rate: %d Hz, target rate: %d Hz (ratio %d:%d)",
            native_rate,
            self.TARGET_RATE,
            self._resample_down,
            self._resample_up,
        )

        try:
            self._stream = self._open_stream(native_rate)
        except sd.PortAudioError:
            # PortAudio's host state can be left corrupted by a previous
            # teardown on macOS/CoreAudio; reinitialize and retry once.
            logger.exception("Failed to open audio stream. Reinitializing PortAudio...")
            sd._terminate()  # noqa: SLF001
            time.sleep(0.5)
            sd._initialize()  # noqa: SLF001
            self._stream = self._open_stream(native_rate)

    def _open_stream(self, native_rate: int) -> sd.InputStream:
        """Open and start an input stream, cleaning up if start fails."""
        stream = sd.InputStream(
            samplerate=native_rate,
            channels=1,
            callback=self._audio_callback,
            dtype="float32",
        )
        try:
            stream.start()
        except sd.PortAudioError:
            stream.close(ignore_errors=True)
            raise
        return stream

    def stop(self) -> None:
        """Stop capturing and process any remaining audio."""
        logger.info("Stopping audio capture...")

        # Graceful stop() rather than abort(): on macOS/CoreAudio, aborting
        # the stream can corrupt PortAudio's internal thread state and make
        # the next stream open fail intermittently.
        try:
            if self._stream:
                try:
                    self._stream.stop()
                finally:
                    self._stream.close(ignore_errors=True)
        finally:
            self._stream = None
            self._drain_queue()
            self._process_and_clear()

        logger.info("Audio capture stopped.")

    def tick(self) -> None:
        """Drain queue, check VAD on tail, process buffer if silence detected."""
        self._drain_queue()

        total_samples = sum(len(chunk) for chunk in self._buffer)
        if total_samples < self.MIN_SAMPLES:
            return

        tail_audio = _extract_tail(self._buffer, self.VAD_TAIL_SAMPLES)
        audio_tensor = torch.from_numpy(tail_audio).float()

        timestamps = get_speech_timestamps(
            audio_tensor, self._vad_model, sampling_rate=self.TARGET_RATE
        )

        if not timestamps:
            self._process_and_clear()

    def _audio_callback(
        self, indata: np.ndarray, _frames: int, _time: Any, status: sd.CallbackFlags
    ) -> None:
        """Called by sounddevice for each audio block (runs in PortAudio C thread)."""
        if status:
            logger.error(status)
        self._chunk_queue.put_nowait(indata.copy())

    def _drain_queue(self) -> None:
        """Move pending chunks from queue into buffer, resampled to TARGET_RATE."""
        needs_resample = self._resample_up != self._resample_down
        while True:
            try:
                chunk = self._chunk_queue.get_nowait()
            except queue.Empty:
                break
            if needs_resample:
                chunk = resample_poly(
                    chunk.flatten(), self._resample_up, self._resample_down
                ).astype(np.float32)
            self._buffer.append(chunk)

    def _process_and_clear(self) -> None:
        """Concatenate buffer, clear it, and pass audio to the callback."""
        if not self._buffer:
            return

        audio_data = np.concatenate(self._buffer, axis=0).flatten()
        self._buffer.clear()
        self.on_audio_ready(audio_data)


def _extract_tail(buffer: list[np.ndarray], num_samples: int) -> np.ndarray:
    """Extract the last `num_samples` from buffer chunks without full concat."""
    tail_chunks: list[np.ndarray] = []
    remaining = num_samples

    for chunk in reversed(buffer):
        if remaining <= 0:
            break
        flat = chunk.flatten()
        if len(flat) >= remaining:
            tail_chunks.append(flat[-remaining:])
            remaining = 0
        else:
            tail_chunks.append(flat)
            remaining -= len(flat)

    tail_chunks.reverse()
    return np.concatenate(tail_chunks, axis=0)


def _drain_all(q: queue.Queue[Any]) -> None:
    """Discard all items from a queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break
