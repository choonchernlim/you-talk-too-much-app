import json
import logging
import os
import warnings
from collections.abc import Generator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import mlx_whisper
import numpy as np
import torch
from pyannote.audio import Pipeline

from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.config import settings
from you_talk_too_much.transcription.formatter import format_conversation
from you_talk_too_much.transcription.speaker_tracker import SpeakerTracker

logger = setup_logger(__name__)

# Suppress FutureWarnings and PyTorch Lightning upgrade warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Suppress all Lightning-related loggers
for logger_name in [
    "lightning",
    "pytorch_lightning",
    "lightning.pytorch.utilities.migration.utils",
]:
    lightning_logger = logging.getLogger(logger_name)
    lightning_logger.setLevel(logging.ERROR)
    lightning_logger.propagate = False

# Thresholds for hallucination detection
NO_SPEECH_PROB_THRESHOLD = 0.7
COMPRESSION_RATIO_THRESHOLD = 2.4


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Context manager to suppress stdout and stderr."""
    with (
        Path(os.devnull).open("w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        yield


class MLXTranscriber:
    """Transcriber using MLX-Whisper and Pyannote for diarization."""

    def __init__(self) -> None:
        """Initialize the MLX Transcriber with whisper and diarization models."""
        self.whisper_model = settings.hf_whisper_model
        self.diarization_model = settings.hf_diarization_model
        self.hf_token = settings.hf_token

        self.pipeline: Pipeline | None = None
        self.device = torch.device("cpu")

        self.speaker_tracker = SpeakerTracker(
            settings.hf_embedding_model, settings.hf_token
        )

        self._initialize_models()

    def reset(self) -> None:
        """Reset the global speaker tracking state."""
        self.speaker_tracker.reset()

    def _initialize_models(self) -> None:
        """Load the diarization pipeline safely and quietly."""
        logger.info(f"Initializing Transcriber ({self.whisper_model})...")

        try:
            logger.info(
                "Initializing Speaker Diarization Pipeline "
                f"({self.diarization_model})..."
            )

            with suppress_output():
                self.pipeline = Pipeline.from_pretrained(
                    self.diarization_model, token=self.hf_token
                )

            if self.pipeline:
                self.device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
                self.pipeline.to(self.device)

        except Exception as e:
            logger.error(f"Error initializing diarization pipeline: {e}")

    def transcribe(self, audio_data: np.ndarray) -> dict[str, Any]:
        """Run MLX whisper transcription."""
        # noinspection PyTypeChecker
        return mlx_whisper.transcribe(
            audio_data, path_or_hf_repo=self.whisper_model, language="en"
        )

    def filter_hallucinations(self, segments: list[dict]) -> list[dict]:
        """Filter out hallucinated segments."""
        valid_segments = []
        for s in segments:
            if s.get("no_speech_prob", 0) > NO_SPEECH_PROB_THRESHOLD:
                continue
            if s.get("compression_ratio", 0) > COMPRESSION_RATIO_THRESHOLD:
                continue
            valid_segments.append(s)
        return valid_segments

    def diarize(self, audio_data: np.ndarray) -> Any:
        """Run Pyannote diarization."""
        if not self.pipeline:
            return None
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        return self.pipeline({"waveform": waveform, "sample_rate": 16000})

    def process(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Process audio data and return (raw_json_str, formatted_conversation)."""
        result = self.transcribe(audio_data)
        raw_json_str = json.dumps(result) + "\n"

        segments = result.get("segments", [])
        segments = self.filter_hallucinations(segments)

        if not segments:
            return raw_json_str, ""

        diarization_output = self.diarize(audio_data)
        local_to_global = self.speaker_tracker.match_speakers(
            audio_data, diarization_output
        )

        formatted_text = format_conversation(
            segments, diarization_output, local_to_global
        )
        return raw_json_str, formatted_text
