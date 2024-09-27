import logging

from log_config import setup_logging
from speech_listener import SpeechListener
from transcriber import VoskTranscriber

setup_logging()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    speech_listener = SpeechListener(VoskTranscriber())
    speech_listener.run()
