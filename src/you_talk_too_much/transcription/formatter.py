from typing import Any

# ANSI escape codes for speaker label colouring
ANSI_RED = "\033[0;31m"
ANSI_RESET = "\033[0m"


def format_conversation(
    segments: list[dict],
    diarization_output: Any,
    local_to_global: dict[str, str],
) -> str:
    """Format transcribed segments into a speaker-labelled conversation string."""
    formatted_lines = []
    current_speaker = None
    current_text_buffer: list[str] = []

    for segment in segments:
        if diarization_output:
            segment_speaker = _get_dominant_speaker(
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
                formatted_lines.append(
                    f"{ANSI_RED}{current_speaker}{ANSI_RESET}: {full_text}"
                )
            current_speaker = segment_speaker
            current_text_buffer = [segment["text"].strip()]

    if current_speaker is not None:
        full_text = " ".join(current_text_buffer)
        formatted_lines.append(f"{ANSI_RED}{current_speaker}{ANSI_RESET}: {full_text}")

    return "\n".join(formatted_lines) + "\n"


def _get_dominant_speaker(
    segment_start: float, segment_end: float, diarization: Any
) -> str:
    """Return the speaker with the most overlap in the given segment range."""
    speaker_durations: dict[str, float] = {}
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
