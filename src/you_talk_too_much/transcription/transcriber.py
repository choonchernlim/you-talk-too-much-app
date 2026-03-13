import json
import os
import sys
import warnings
from typing import Any, Dict, cast

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

# Suppress all Lightning-related loggers
for name in ["lightning", "pytorch_lightning", "lightning.pytorch.utilities.migration.utils"]:
    l = logging.getLogger(name)
    l.setLevel(logging.ERROR)
    l.propagate = False

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

        _stdout = sys.stdout
        _stderr = sys.stderr

        try:
            logger.info(
                f"Initializing Speaker Diarization Pipeline ({self.diarization_model})..."
            )

            # Suppress hardcoded Pyannote/HF/Lightning prints
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            self.pipeline = Pipeline.from_pretrained(
                self.diarization_model, token=self.hf_token
            )

            if self.pipeline:
                self.device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
                self.pipeline.to(self.device)

            # Briefly restore to log the next step
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = _stdout
            sys.stderr = _stderr

            logger.info(
                f"Initializing Embedding Model ({self.embedding_model_name})..."
            )

            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            self.embedding_model = Model.from_pretrained(
                self.embedding_model_name, token=self.hf_token
            )
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

        finally:
            if sys.stdout is not _stdout:
                sys.stdout.close()
                sys.stdout = _stdout
            if sys.stderr is not _stderr:
                sys.stderr.close()
                sys.stderr = _stderr

    def transcribe(self, audio_data: np.ndarray) -> Dict[str, Any]:
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

    def match_speakers(
            self, audio_data: np.ndarray, diarization_output: Any
    ) -> dict[str, str]:
        """Match local speakers to global speakers using embeddings."""
        local_to_global = {}
        if not self.embedding_model or not diarization_output:
            return local_to_global

        # Handle pyannote-audio 4.x DiarizeOutput object
        annotation = getattr(diarization_output, "speaker_diarization", diarization_output)

        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        inference = Inference(self.embedding_model, window="whole")
        duration = audio_data.shape[0] / 16000

        for local_speaker in annotation.labels():
            embeddings = []
            for turn, _, speaker_label in annotation.itertracks(
                    yield_label=True
            ):
                if speaker_label != local_speaker:
                    continue
                # Ignore short segments
                if turn.end - turn.start < 1.2:
                    continue
                try:
                    # Clamp turn to waveform duration
                    clamped_turn = turn
                    if turn.end > duration:
                        from pyannote.core import Segment
                        clamped_turn = Segment(turn.start, min(turn.end, duration))

                    if clamped_turn.end - clamped_turn.start < 0.1:
                        continue

                    emb = inference.crop(
                        {"waveform": waveform, "sample_rate": 16000}, clamped_turn
                    )
                    embeddings.append(emb)
                except Exception as e:
                    logger.debug(f"Embedding error (clamped {turn} to {duration}s): {e}")

            if not embeddings:
                continue

            stacked_embeddings = np.vstack(embeddings)
            if stacked_embeddings.size == 0:
                continue

            avg_embedding = np.mean(stacked_embeddings, axis=0)

            if not self.global_speakers:
                global_id = f"SPEAKER_{self.speaker_counter:02d}"
                self.speaker_counter += 1
                self.global_speakers[global_id] = avg_embedding
                local_to_global[local_speaker] = global_id
            else:
                global_ids = list(self.global_speakers.keys())
                # Ensure global_embs is correctly stacked
                global_embs = np.vstack(list(self.global_speakers.values()))
                if global_embs.size == 0:
                    continue
                
                distances = cdist([avg_embedding], global_embs, metric="cosine")[0]
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]

                if min_dist < 0.55:
                    matched_id = global_ids[min_idx]
                    local_to_global[local_speaker] = matched_id
                    if min_dist < 0.25:
                        self.global_speakers[matched_id] = (
                                0.9 * self.global_speakers[matched_id] + 0.1 * avg_embedding
                        )
                else:
                    global_id = f"SPEAKER_{self.speaker_counter:02d}"
                    self.speaker_counter += 1
                    self.global_speakers[global_id] = avg_embedding
                    local_to_global[local_speaker] = global_id

        return local_to_global

    def _get_dominant_speaker(
            self, segment_start: float, segment_end: float, diarization: Any
    ) -> str:
        """Finds the dominant speaker during a given segment."""
        speaker_durations: dict[str, float] = {}
        # Handle pyannote-audio 4.x DiarizeOutput object
        annotation = getattr(diarization, "speaker_diarization", diarization)

        for turn, _, speaker_id in annotation.itertracks(yield_label=True):
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

    def format_conversation(
            self,
            segments: list[dict],
            diarization_output: Any,
            local_to_global: dict[str, str],
    ) -> str:
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

        formatted_text = self.format_conversation(
            segments, diarization_output, local_to_global
        )
        return raw_json_str, formatted_text
