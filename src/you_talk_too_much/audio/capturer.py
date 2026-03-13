import time
from collections.abc import Callable
from threading import Event
from typing import Any

import numpy as np
import sounddevice as sd
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)


class AudioCapturer:
    """Audio Capturer class using sounddevice."""

    RATE = 16000

    def __init__(self, on_audio_ready: Callable[[np.ndarray], None]) -> None:
        """Initialize the audio capturer."""
        logger.info("Initializing Audio Capturer...")

        self.on_audio_ready = on_audio_ready

        # Buffer to store batched audio data as list of numpy arrays
        self.buffer: list[np.ndarray] = []

    def capture_audio(self, stop_event: Event) -> None:
        """Start capturing audio in a loop until stop_event is set."""
        logger.info("Starting audio capture...")

        self.buffer.clear()

        def _audio_callback(
            indata: np.ndarray, _frames: int, _time: Any, status: sd.CallbackFlags
        ) -> None:
            """This is called for each audio block by sounddevice."""
            if status:
                logger.error(status)
            self.buffer.append(indata.copy())

        with sd.InputStream(
            samplerate=self.RATE,
            channels=1,
            callback=_audio_callback,
            dtype="float32",
        ):
            while not stop_event.is_set():
                stop_event.wait(0.1)

        logger.info("Stopping audio capture...")

    def batch_process_buffer(
        self, stop_event: Event, check_interval: float = 2.0
    ) -> None:
        """Process the audio buffer dynamically based on VAD."""
        logger.info("Starting VAD-based batch process buffer...")

        model = load_silero_vad()

        while not stop_event.is_set():
            time.sleep(check_interval)

            if not self.buffer:
                continue

            # Check total accumulated audio
            total_samples = sum(len(chunk) for chunk in self.buffer)
            # Require at least 5 seconds of audio
            if total_samples < self.RATE * 5:
                continue

            # Concatenate safely for VAD check
            current_audio = np.concatenate(self.buffer, axis=0).flatten()
            audio_tensor = torch.from_numpy(current_audio).float()

            # Check if last 1.5 seconds is silence
            # 1.5 seconds * 16000 = 24000 samples
            last_samples = audio_tensor[-24000:]

            # silero-vad expects 1D float32 tensor
            timestamps = get_speech_timestamps(
                last_samples, model, sampling_rate=self.RATE
            )

            if not timestamps:
                # No speech in the last 1.5 seconds, we have a pause! Process the buffer.
                self.process_buffer()

        logger.info("Stopping batch process buffer...")

        # Process the remaining audio in the buffer
        self.process_buffer()

    def process_buffer(self) -> None:
        """Process the current audio buffer and clear it atomically."""
        if self.buffer:
            # Copy references and clear atomically to prevent losing new incoming chunks
            current_chunks = self.buffer[:]
            self.buffer.clear()

            # Concatenate all recorded chunks
            audio_data = np.concatenate(current_chunks, axis=0).flatten()

            # Pass the audio data via the callback
            self.on_audio_ready(audio_data)
