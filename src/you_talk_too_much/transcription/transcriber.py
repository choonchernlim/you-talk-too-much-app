import json
import os
import sys
import warnings
from typing import Any

import mlx_whisper
import numpy as np
import torch
from pyannote.audio import Inference, Model, Pipeline
from scipy.spatial.distance import cdist

from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.config import settings

logger = setup_logger(__name__)

# Suppress FutureWarnings and PyTorch Lightning upgrade warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Thresholds for hallucination detection
NO_SPEECH_PROB_THRESHOLD = 0.7
COMPRESSION_RATIO_THRESHOLD = 2.4


class MLXTranscriber:
    """Transcriber using MLX-Whisper and Pyannote for diarization."""

    def __init__(self) -> None:
        """Initialize the MLX Transcriber with whisper and diarization models."""
        self.whisper_model = settings.hf_whisper_model
        self.diarization_model = settings.hf_diarization_model
        self.embedding_model_name = settings.hf_embedding_model
        self.hf_token = settings.hf_token

        self.pipeline = None
        self.embedding_model = None
        self.device = torch.device("cpu")

        # Global speaker tracking
        self.global_speakers: dict[str, np.ndarray] = {}
        self.speaker_counter = 0

        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load ML models safely and quietly."""
        logger.info(f"Initializing Transcriber ({self.whisper_model})...")

        # PyTorch 2.6 default weights_only=True breaks pyannote model loading
        _original_load = torch.load
        _stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        try:
            torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

            logger.info(f"Initializing Speaker Diarization Pipeline ({self.diarization_model})...")
            self.pipeline = Pipeline.from_pretrained(self.diarization_model)

            if self.pipeline:
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                self.pipeline.to(self.device)

                sys.stdout.close()
                sys.stdout = _stdout
                sys.stdout = open(os.devnull, "w")
            else:
                sys.stdout.close()
                sys.stdout = _stdout
                sys.stdout = open(os.devnull, "w")

            try:
                logger.info(f"Initializing Embedding Model ({self.embedding_model_name})...")
                self.embedding_model = Model.from_pretrained(
                    self.embedding_model_name, use_auth_token=self.hf_token
                )
                self.embedding_model.to(self.device)
                self.embedding_model.eval()

                sys.stdout.close()
                sys.stdout = _stdout
            except Exception as e:
                sys.stdout.close()
                sys.stdout = _stdout
        finally:
            torch.load = _original_load
            if sys.stdout is not _stdout:
                sys.stdout.close()
                sys.stdout = _stdout

    def transcribe(self, audio_data: np.ndarray) -> dict:
        """Run MLX whisper transcription."""
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

    def match_speakers(self, audio_data: np.ndarray, diarization_output: Any) -> dict[str, str]:
        """Match local speakers to global speakers using embeddings."""
        local_to_global = {}
        if not self.embedding_model or not diarization_output:
            return local_to_global

        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        inference = Inference(self.embedding_model, window="whole")

        for local_speaker in diarization_output.labels():
            embeddings = []
            for turn, _, speaker_label in diarization_output.itertracks(yield_label=True):
                if speaker_label != local_speaker:
                    continue
                # Ignore short segments
                if turn.end - turn.start < 1.2:
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

                if min_dist < 0.55:
                    matched_id = global_ids[min_idx]
                    local_to_global[local_speaker] = matched_id
                    if min_dist < 0.25:
                        self.global_speakers[matched_id] = 0.9 * self.global_speakers[matched_id] + 0.1 * avg_embedding
                else:
                    global_id = f"SPEAKER_{self.speaker_counter:02d}"
                    self.speaker_counter += 1
                    self.global_speakers[global_id] = avg_embedding
                    local_to_global[local_speaker] = global_id

        return local_to_global

    def _get_dominant_speaker(self, segment_start: float, segment_end: float, diarization: Any) -> str:
        """Finds the dominant speaker during a given segment."""
        speaker_durations: dict[str, float] = {}
        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            if overlap_duration > 0:
                speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0.0) + overlap_duration

        if speaker_durations:
            return max(speaker_durations, key=lambda k: speaker_durations[k])
        return "SPEAKER_UNKNOWN"

    def format_conversation(self, segments: list[dict], diarization_output: Any, local_to_global: dict[str, str]) -> str:
        """Format the transcribed segments with speaker labels."""
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

        return "\n".join(formatted_lines) + "\n"

    def process(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Process audio data and return (raw_json_str, formatted_conversation)."""
        result = self.transcribe(audio_data)
        raw_json_str = json.dumps(result) + "\n"

        segments = result.get("segments", [])
        segments = self.filter_hallucinations(segments)

        if not segments:
            return raw_json_str, ""

        diarization_output = self.diarize(audio_data)
        local_to_global = self.match_speakers(audio_data, diarization_output)

        formatted_text = self.format_conversation(segments, diarization_output, local_to_global)
        return raw_json_str, formatted_text
