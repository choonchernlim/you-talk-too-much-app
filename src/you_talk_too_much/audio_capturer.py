import time
from threading import Event
from typing import Any

import numpy as np
import sounddevice as sd

from you_talk_too_much.log_config import setup_logger
from you_talk_too_much.transcriber import Transcriber

logger = setup_logger(__name__)


class AudioCapturer:
    """Audio Capturer class using sounddevice."""

    RATE = 16000

    def __init__(self, transcriber: Transcriber) -> None:
        """Initialize the audio capturer."""
        logger.info("Initializing Audio Capturer...")

        self.transcriber = transcriber

        # Buffer to store batched audio data as list of numpy arrays
        self.buffer: list[np.ndarray] = []

    def capture_audio(self, stop_event: Event) -> None:
        """Start capturing audio in a loop until stop_event is set."""
        logger.info("Starting audio capture...")

        # create new transcript and clear the audio buffer
        self.transcriber.create_new_transcript_directory()
        self.buffer.clear()

        def _audio_callback(
            indata: np.ndarray, _frames: int, _time: Any, status: sd.CallbackFlags
        ) -> None:
            """This is called for each audio block by sounddevice."""
            if status:
                logger.error(status)
            self.buffer.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.RATE,
                channels=1,
                callback=_audio_callback,
                dtype="float32",
            ):
                while not stop_event.is_set():
                    stop_event.wait(0.1)
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            stop_event.set()

        logger.info("Stopping audio capture...")

    def batch_process_buffer(
        self, stop_event: Event, batch_duration_in_sec: int = 30
    ) -> None:
        """Process the audio buffer in batches until stop_event is set."""
        logger.info("Starting batch process buffer...")

        while not stop_event.is_set():
            # Wait for enough audio to accumulate
            time.sleep(batch_duration_in_sec)

            self.process_buffer()

        logger.info("Stopping batch process buffer...")

        # Process the remaining audio in the buffer
        self.process_buffer()

    def process_buffer(self) -> None:
        """Process the current audio buffer and clear it."""
        if self.buffer:
            # Concatenate all recorded chunks
            audio_data = np.concatenate(self.buffer, axis=0).flatten()

            # Clear buffer after batching
            self.buffer.clear()

            conversation_text = self.transcriber.run(audio_data)

            # Print the conversation text to console
            if conversation_text:
                logger.info(f"Conversation: {conversation_text}")
