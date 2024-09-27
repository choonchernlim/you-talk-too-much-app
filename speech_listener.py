import logging

import pyaudio

from transcriber import Transcriber

# Install portaudio before running pip install
# https://stackoverflow.com/questions/33513522/when-installing-pyaudio-pip-cannot-find-portaudio-h-in-usr-local-include

logger = logging.getLogger(__name__)


class SpeechListener:
    SAMPLE_RATE = 16000
    FRAMES_PER_BUFFER = 8192

    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber
        self.p = pyaudio.PyAudio()

        # Open the microphone stream
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.FRAMES_PER_BUFFER,
        )

    def run(self) -> None:
        logger.info("=====================================================")
        logger.info("Listening for speech. Press CTRL+C to stop.")
        logger.info("=====================================================")

        try:
            while True:
                if self.stream.get_read_available():
                    frame = self.stream.read(self.FRAMES_PER_BUFFER)
                    self.transcriber.stream_handler(frame)
                else:
                    self.transcriber.no_stream_handler()

        # capture CTRL+C to terminate
        except KeyboardInterrupt:
            self.transcriber.no_stream_handler()

            logger.info("=====================================================")
            logger.info("Termination keyword detected. Stopping...")
            logger.info("=====================================================")

            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
