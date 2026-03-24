import queue
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)


class AudioCapturer:
    """Audio Capturer using sounddevice with a single-threaded tick pattern."""

    RATE = 16000
    MIN_SAMPLES = RATE * 5  # 5 seconds minimum before VAD check
    VAD_TAIL_SAMPLES = 24000  # 1.5 seconds for silence detection

    def __init__(self, on_audio_ready: Callable[[np.ndarray], None]) -> None:
        """Initialize the audio capturer."""
        logger.info("Initializing Audio Capturer...")

        self.on_audio_ready = on_audio_ready
        self._chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._buffer: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._vad_model = load_silero_vad()

    def start(self) -> None:
        """Start capturing audio by opening the sounddevice stream."""
        logger.info("Starting audio capture...")

        self._buffer.clear()
        _drain_all(self._chunk_queue)

        self._stream = sd.InputStream(
            samplerate=self.RATE,
            channels=1,
            callback=self._audio_callback,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing and process any remaining audio."""
        logger.info("Stopping audio capture...")

        if self._stream:
            self._stream.abort()
            self._stream.close()
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
            audio_tensor, self._vad_model, sampling_rate=self.RATE
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
        """Move all pending chunks from the queue into the local buffer."""
        while True:
            try:
                self._buffer.append(self._chunk_queue.get_nowait())
            except queue.Empty:
                break

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
