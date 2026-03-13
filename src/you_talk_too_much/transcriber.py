import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import mlx_whisper
import numpy as np
import torch
from pyannote.audio import Inference, Model, Pipeline
from scipy.spatial.distance import cdist

from you_talk_too_much.log_config import setup_logger
from you_talk_too_much.utils import append_file

logger = setup_logger(__name__)

# Suppress FutureWarnings and PyTorch Lightning upgrade warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

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
        self.embedding_model_name = os.getenv("HF_EMBEDDING_MODEL")
        self.hf_token = os.getenv("HF_TOKEN")

        # PyTorch 2.6 default weights_only=True breaks pyannote model loading
        _original_load = torch.load

        # Suppress hardcoded prints inside pyannote
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        try:
            torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

            self.pipeline = Pipeline.from_pretrained(self.diarization_model)

            if self.pipeline:
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.pipeline.to(self.device)

                # Temporarily restore stdout for our own logs
                sys.stdout.close()
                sys.stdout = _stdout
                logger.info(f"Diarization pipeline loaded on {self.device}")
                sys.stdout = open(os.devnull, "w")
            else:
                self.device = torch.device("cpu")
                sys.stdout.close()
                sys.stdout = _stdout
                logger.error("Failed to load diarization pipeline.")
                sys.stdout = open(os.devnull, "w")

            try:
                self.embedding_model = Model.from_pretrained(
                    self.embedding_model_name, use_auth_token=self.hf_token
                )
                self.embedding_model.to(self.device)
                self.embedding_model.eval()

                sys.stdout.close()
                sys.stdout = _stdout
                logger.info(f"Embedding model loaded on {self.device}")
            except Exception as e:
                sys.stdout.close()
                sys.stdout = _stdout
                logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None

        finally:
            torch.load = _original_load
            if sys.stdout is not _stdout:
                sys.stdout.close()
                sys.stdout = _stdout

        # Global speaker tracking
        self.global_speakers: dict[str, np.ndarray] = {}
        self.speaker_counter = 0

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

        # Cross-chunk speaker matching
        local_to_global = {}
        if self.embedding_model and diarization_output:
            inference = Inference(self.embedding_model, window="whole")
            for local_speaker in diarization_output.labels():
                embeddings = []
                for turn, _, speaker_label in diarization_output.itertracks(yield_label=True):
                    if speaker_label != local_speaker:
                        continue
                    if turn.end - turn.start < 0.5:
                        continue
                    try:
                        emb = inference.crop({"waveform": waveform, "sample_rate": 16000}, turn)
                        embeddings.append(emb)
                    except Exception as e:
                        logger.error(f"Embedding error: {e}")

                if not embeddings:
                    continue

                avg_embedding = np.mean(np.vstack(embeddings), axis=0)

                if not self.global_speakers:
                    global_id = f"SPEAKER_{self.speaker_counter:02d}"
                    self.speaker_counter += 1
                    self.global_speakers[global_id] = avg_embedding
                    local_to_global[local_speaker] = global_id
                else:
                    global_ids = list(self.global_speakers.keys())
                    global_embs = np.vstack(list(self.global_speakers.values()))
                    distances = cdist([avg_embedding], global_embs, metric="cosine")[0]
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]

                    if min_dist < 0.3: # Match threshold
                        matched_id = global_ids[min_idx]
                        local_to_global[local_speaker] = matched_id
                        self.global_speakers[matched_id] = 0.9 * self.global_speakers[matched_id] + 0.1 * avg_embedding
                    else:
                        global_id = f"SPEAKER_{self.speaker_counter:02d}"
                        self.speaker_counter += 1
                        self.global_speakers[global_id] = avg_embedding
                        local_to_global[local_speaker] = global_id

        formatted_lines = []
        current_speaker = None
        current_text_buffer = []

        for segment in segments:
            if diarization_output:
                segment_speaker = self._get_dominant_speaker(
                    segment["start"], segment["end"], diarization_output
                )
                segment_speaker = local_to_global.get(segment_speaker, segment_speaker)
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
