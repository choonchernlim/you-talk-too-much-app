import time
from threading import Event

import pyaudio

from app.log_config import setup_logger
from app.transcriber import Transcriber

logger = setup_logger(__name__)


class AudioCapturer:
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms chunks

    def __init__(self, transcriber: Transcriber):
        logger.info('Initializing Audio Capturer...')

        self.transcriber = transcriber

        # Buffer to store batched audio data
        self.buffer = []

    def capture_audio(self, stop_event: Event):
        logger.info('Starting audio capture...')

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        while not stop_event.is_set():
            data = stream.read(self.CHUNK)
            self.buffer.append(data)

        logger.info('Stopping audio capture...')
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def batch_process_buffer(self, stop_event: Event, batch_duration_in_sec=2):
        logger.info('Starting batch process buffer...')

        while not stop_event.is_set():
            # Wait for enough audio to accumulate
            time.sleep(batch_duration_in_sec)

            self.process_buffer()

        logger.info('Stopping batch process buffer...')

        # Process the remaining audio in the buffer
        self.process_buffer()

        # Print a newline
        print()

    def process_buffer(self):
        if self.buffer:
            audio_data = b''.join(self.buffer)

            # Clear buffer after batching
            self.buffer.clear()

            conversation_text = self.transcriber.run(audio_data)

            # Print the conversation text to console
            print(conversation_text, end='', flush=True)
