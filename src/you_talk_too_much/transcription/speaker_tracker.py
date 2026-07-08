from typing import Any

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pyannote.core import Segment
from scipy.spatial.distance import cdist

from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.common.output import suppress_output

logger = setup_logger(__name__)

# Thresholds for speaker matching
MIN_TURN_DURATION = 1.2
MIN_CLAMPED_DURATION = 0.1
COSINE_SIMILARITY_MATCH_THRESHOLD = 0.72
COSINE_SIMILARITY_UPDATE_THRESHOLD = 0.50
SPEAKER_EMBEDDING_UPDATE_WEIGHT = 0.1


class SpeakerTracker:
    """Tracks global speaker identities across audio chunks using embeddings."""

    def __init__(self, embedding_model_name: str, hf_token: str) -> None:
        """Load the embedding model and initialise speaker state."""
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        with suppress_output():
            self.embedding_model: Model | None = Model.from_pretrained(
                embedding_model_name, token=hf_token
            )

        if self.embedding_model:
            self.embedding_model.to(self.device)
            self.embedding_model.eval()

        self.global_speakers: dict[str, np.ndarray] = {}
        self.speaker_counter = 0

    def reset(self) -> None:
        """Reset the global speaker tracking state."""
        self.global_speakers.clear()
        self.speaker_counter = 0

    def match_speakers(
        self, audio_data: np.ndarray, diarization_output: Any
    ) -> dict[str, str]:
        """Match local speakers to global speakers using embeddings."""
        local_to_global: dict[str, str] = {}
        if not self.embedding_model or not diarization_output:
            return local_to_global

        annotation = getattr(
            diarization_output, "speaker_diarization", diarization_output
        )
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        inference = Inference(self.embedding_model, window="whole")
        duration = audio_data.shape[0] / 16000

        for local_speaker in annotation.labels():
            embeddings = self._get_speaker_embeddings(
                annotation, local_speaker, waveform, duration, inference
            )
            if not embeddings:
                continue

            stacked = np.vstack(embeddings)
            if stacked.size == 0:
                continue

            avg_embedding = np.mean(stacked, axis=0)
            self._assign_global_id(local_speaker, avg_embedding, local_to_global)

        return local_to_global

    def _get_speaker_embeddings(
        self,
        annotation: Any,
        local_speaker: str,
        waveform: torch.Tensor,
        duration: float,
        inference: Inference,
    ) -> list[np.ndarray]:
        """Extract embeddings for a specific local speaker."""
        embeddings = _collect_embeddings(
            annotation,
            local_speaker,
            waveform,
            duration,
            inference,
            min_turn_duration=MIN_TURN_DURATION,
        )

        # Fallback: retry without the minimum turn duration filter
        if not embeddings:
            embeddings = _collect_embeddings(
                annotation,
                local_speaker,
                waveform,
                duration,
                inference,
                min_turn_duration=0.0,
            )

        return embeddings

    def _assign_global_id(
        self,
        local_speaker: str,
        avg_embedding: np.ndarray,
        local_to_global: dict[str, str],
    ) -> None:
        """Assign or match a global speaker ID to a local speaker."""
        if not self.global_speakers:
            global_id = f"SPEAKER_{self.speaker_counter:02d}"
            self.speaker_counter += 1
            self.global_speakers[global_id] = avg_embedding
            local_to_global[local_speaker] = global_id
            return

        global_ids = list(self.global_speakers.keys())
        global_embs = np.vstack(list(self.global_speakers.values()))
        if global_embs.size == 0:
            return

        distances = cdist([avg_embedding], global_embs, metric="cosine")[0]
        min_idx = int(np.argmin(distances))
        min_dist = distances[min_idx]

        if min_dist < COSINE_SIMILARITY_MATCH_THRESHOLD:
            matched_id = global_ids[min_idx]
            local_to_global[local_speaker] = matched_id
            if min_dist < COSINE_SIMILARITY_UPDATE_THRESHOLD:
                self.global_speakers[matched_id] = (
                    1 - SPEAKER_EMBEDDING_UPDATE_WEIGHT
                ) * self.global_speakers[
                    matched_id
                ] + SPEAKER_EMBEDDING_UPDATE_WEIGHT * avg_embedding
        else:
            global_id = f"SPEAKER_{self.speaker_counter:02d}"
            self.speaker_counter += 1
            self.global_speakers[global_id] = avg_embedding
            local_to_global[local_speaker] = global_id


def _collect_embeddings(
    annotation: Any,
    local_speaker: str,
    waveform: torch.Tensor,
    duration: float,
    inference: Inference,
    min_turn_duration: float,
) -> list[np.ndarray]:
    """Collect speaker embeddings for turns that meet the duration threshold."""
    embeddings = []
    for turn, _, speaker_label in annotation.itertracks(yield_label=True):
        if speaker_label != local_speaker:
            continue
        if min_turn_duration > 0 and turn.end - turn.start < min_turn_duration:
            continue
        try:
            clamped = Segment(turn.start, min(turn.end, duration))
            if clamped.end - clamped.start < MIN_CLAMPED_DURATION:
                continue
            emb = inference.crop({"waveform": waveform, "sample_rate": 16000}, clamped)
            embeddings.append(emb)
        except Exception as e:
            logger.debug(f"Embedding error (turn {turn}, duration {duration}s): {e}")
    return embeddings
