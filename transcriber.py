import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

import vosk
from dotenv import load_dotenv

from conversation import VoskConversationParser

logger = logging.getLogger(__name__)

load_dotenv()


class Transcriber(ABC):
    @abstractmethod
    def get_conversation_file_path(self) -> str:
        pass

    @abstractmethod
    def get_raw_file_path(self) -> str:
        pass

    @abstractmethod
    def stream_handler(self, frame):
        pass

    @abstractmethod
    def no_stream_handler(self):
        pass


# https://medium.com/@nimritakoul01/offline-speech-to-text-in-python-f5d6454ecd02
class VoskTranscriber(Transcriber):
    def __init__(self):
        self.model = vosk.Model(model_path=os.getenv('VOSK_MODEL_PATH'))

        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        self.rec.SetWords(True)

        self.datetime_suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.conversation_parser = VoskConversationParser(
            conversation_file_path=self.get_conversation_file_path(),
            raw_file_path=self.get_raw_file_path(),
        )

    def get_conversation_file_path(self) -> str:
        return f'out/{self.datetime_suffix}-conversation.txt'

    def get_raw_file_path(self) -> str:
        return f'out/{self.datetime_suffix}-raw.txt'

    def stream_handler(self, frame):
        if self.rec.AcceptWaveform(frame):
            # Parse the JSON result and get the recognized text
            words = json.loads(self.rec.Result()).get('result', [])
            self.conversation_parser.conversation_parser(words)

    def no_stream_handler(self):
        self.conversation_parser.conversation_parser()
