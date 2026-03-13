import json
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import mlx_whisper
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications

from you_talk_too_much.log_config import setup_logger
from you_talk_too_much.utils import append_file

# Fix for PyTorch 2.6+ weight loading security changes
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(
        [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
    )

logger = setup_logger(__name__)

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Thresholds for hallucination detection
NO_SPEECH_PROB_THRESHOLD = 0.7
COMPRESSION_RATIO_THRESHOLD = 0.5555555555555556


class Transcriber(ABC):
    """Abstract base class for transcribers."""

    def __init__(self) -> None:
        """Initialize Transcriber."""
        super().__init__()

        self.formatted_datetime: str = ""
        self.out_dir: str = ""

    def create_new_transcript_directory(self) -> None:
        """Create a new directory for storing transcripts."""
        logger.info("Creating new transcript directory...")

        self.formatted_datetime = datetime.now().strftime("%Y-%m-%d %p %I:%M")
        self.out_dir = f"transcripts/{self.formatted_datetime}"

        # create folder if not exists
        os.makedirs(self.out_dir, exist_ok=True)

    def get_formatted_datetime(self) -> str:
        """Return the formatted datetime of the current session."""
        return self.formatted_datetime

    def get_conversation_file_path(self) -> str:
        """Return the path to the conversation text file."""
        assert self.out_dir != ""

        return f"{self.out_dir}/conversation.txt"

    def get_raw_file_path(self) -> str:
        """Return the path to the raw transcription JSONL file."""
        assert self.out_dir != ""

        return f"{self.out_dir}/raw.jsonl"

    @abstractmethod
    def run(self, audio_data: np.ndarray) -> str:
        """Process audio data and return transcribed text."""
        pass


class MLXTranscriber(Transcriber):
    """Transcriber using MLX-Whisper and Pyannote for diarization."""

    def __init__(self) -> None:
        """Initialize the MLX Transcriber with whisper and diarization models."""
        super().__init__()

        logger.info("Initializing MLX Transcriber...")

        self.whisper_model = os.getenv("HF_WHISPER_MODEL")
        self.diarization_model = os.getenv("HF_DIARIZATION_MODEL")
        self.hf_token = os.getenv("HF_TOKEN")

        self.pipeline = Pipeline.from_pretrained(self.diarization_model)

        if self.pipeline:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.pipeline.to(device)
            logger.info(f"Diarization pipeline loaded on {device}")
        else:
            logger.error("Failed to load diarization pipeline.")

    def _get_dominant_speaker(
        self, segment_start: float, segment_end: float, diarization: Any
    ) -> str:
        """Finds the dominant speaker during a given segment."""
        speaker_durations: dict[str, float] = {}
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            if overlap_duration > 0:
                speaker_durations[speaker_id] = (
                    speaker_durations.get(speaker_id, 0.0) + overlap_duration
                )

        if speaker_durations:
            return max(speaker_durations, key=lambda k: speaker_durations[k])
        return "SPEAKER_UNKNOWN"

    def run(self, audio_data: np.ndarray) -> str:
        """Transcribe and diarize the given audio data."""
        # Process audio with MLX Whisper
        result = mlx_whisper.transcribe(
            audio_data, path_or_hf_repo=self.whisper_model, language="en"
        )

        append_file(self.get_raw_file_path(), json.dumps(result) + "\n")

        segments = result.get("segments", [])

        # detect hallucinated text
        if not segments or all(
            s.get("no_speech_prob", 0) > NO_SPEECH_PROB_THRESHOLD
            or s.get("compression_ratio", 0) == COMPRESSION_RATIO_THRESHOLD
            for s in segments
        ):
            return ""

        # Diarization
        if self.pipeline:
            # Pyannote pipeline expects waveform as (channels, samples)
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
            diarization_output = self.pipeline(
                {"waveform": waveform, "sample_rate": 16000}
            )
        else:
            diarization_output = None

        formatted_lines = []
        current_speaker = None
        current_text_buffer = []

        for segment in segments:
            if diarization_output:
                segment_speaker = self._get_dominant_speaker(
                    segment["start"], segment["end"], diarization_output
                )
            else:
                segment_speaker = "UNKNOWN"

            if segment_speaker == current_speaker:
                current_text_buffer.append(segment["text"].strip())
            else:
                if current_speaker is not None:
                    full_text = " ".join(current_text_buffer)
                    formatted_lines.append(f"{current_speaker}: {full_text}")

                current_speaker = segment_speaker
                current_text_buffer = [segment["text"].strip()]

        if current_speaker is not None:
            full_text = " ".join(current_text_buffer)
            formatted_lines.append(f"{current_speaker}: {full_text}")

        final_text = "\n".join(formatted_lines) + "\n"
        append_file(self.get_conversation_file_path(), final_text)

        return final_text
