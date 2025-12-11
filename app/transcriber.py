import json
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import whisper

from app.log_config import setup_logger
from app.utils import append_file

logger = setup_logger(__name__)

# Suppress FutureWarnings thrown by Whisper
warnings.simplefilter(action='ignore', category=FutureWarning)


class Transcriber(ABC):
    def __init__(self):
        super().__init__()

        self.formatted_datetime = None
        self.out_dir = None

    def create_new_transcript_directory(self):
        logger.info('Creating new transcript directory...')

        self.formatted_datetime = datetime.now().strftime('%Y-%m-%d %p %I:%M')
        self.out_dir = f'transcripts/{self.formatted_datetime}'

        # create folder out/[formatted_datetime] if not exists
        os.makedirs(self.out_dir, exist_ok=True)

    def get_formatted_datetime(self) -> str:
        return self.formatted_datetime

    def get_conversation_file_path(self) -> str:
        assert self.out_dir is not None

        return f'{self.out_dir}/conversation.txt'

    def get_raw_file_path(self) -> str:
        assert self.out_dir is not None

        return f'{self.out_dir}/raw.jsonl'

    @abstractmethod
    def run(self, audio_data) -> str:
        pass


class WhisperTranscriber(Transcriber):
    def __init__(self):
        super().__init__()

        logger.info('Initializing Whisper Transcriber...')

        # Load Whisper model
        self.model = whisper.load_model('large-v3')  # base

    def run(self, audio_data) -> str:
        # Process audio with Whisper model
        normalized_audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.model.transcribe(normalized_audio_data, language='en', fp16=False)

        append_file(self.get_raw_file_path(), json.dumps(result) + '\n')

        text = result['text']
        segments = result['segments']

        # detect hallucinated text and return empty string instead
        if (not segments or
                all(s['no_speech_prob'] > 0.7 or s['compression_ratio'] == 0.5555555555555556
                    for s in segments)):
            return ''

        append_file(self.get_conversation_file_path(), text.strip() + '\n')

        return text
