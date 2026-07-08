from you_talk_too_much.transcription.formatter import (
    _get_dominant_speaker,
    format_conversation,
)


class FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class FakeAnnotation:
    """Mimics pyannote Annotation.itertracks(yield_label=True)."""

    def __init__(self, tracks: list[tuple[FakeTurn, str, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label
        yield from self._tracks


def _strip_ansi(text: str) -> str:
    return text.replace("\033[0;31m", "").replace("\033[0m", "")


class TestFormatConversation:
    def test_labels_segments_with_dominant_speaker(self) -> None:
        diarization = FakeAnnotation(
            [
                (FakeTurn(0.0, 2.0), "A", "SPEAKER_A"),
                (FakeTurn(2.0, 4.0), "B", "SPEAKER_B"),
            ]
        )
        segments = [
            {"start": 0.0, "end": 2.0, "text": " hello "},
            {"start": 2.0, "end": 4.0, "text": " world "},
        ]

        result = format_conversation(segments, diarization, {})

        assert _strip_ansi(result) == "SPEAKER_A: hello\nSPEAKER_B: world\n"

    def test_merges_consecutive_segments_of_same_speaker(self) -> None:
        diarization = FakeAnnotation([(FakeTurn(0.0, 10.0), "A", "SPEAKER_A")])
        segments = [
            {"start": 0.0, "end": 2.0, "text": "one"},
            {"start": 2.0, "end": 4.0, "text": "two"},
        ]

        result = format_conversation(segments, diarization, {})

        assert _strip_ansi(result) == "SPEAKER_A: one two\n"

    def test_maps_local_speakers_to_global_ids(self) -> None:
        diarization = FakeAnnotation([(FakeTurn(0.0, 2.0), "A", "SPEAKER_00")])
        segments = [{"start": 0.0, "end": 2.0, "text": "hi"}]

        result = format_conversation(
            segments, diarization, {"SPEAKER_00": "SPEAKER_07"}
        )

        assert _strip_ansi(result) == "SPEAKER_07: hi\n"

    def test_uses_unknown_when_no_diarization(self) -> None:
        segments = [{"start": 0.0, "end": 2.0, "text": "hi"}]

        result = format_conversation(segments, None, {})

        assert _strip_ansi(result) == "UNKNOWN: hi\n"

    def test_empty_segments_produce_only_newline(self) -> None:
        result = format_conversation([], None, {})

        assert result == "\n"


class TestGetDominantSpeaker:
    def test_returns_speaker_with_most_overlap(self) -> None:
        diarization = FakeAnnotation(
            [
                (FakeTurn(0.0, 1.0), "A", "SPEAKER_A"),
                (FakeTurn(1.0, 5.0), "B", "SPEAKER_B"),
            ]
        )

        assert _get_dominant_speaker(0.0, 5.0, diarization) == "SPEAKER_B"

    def test_returns_unknown_when_no_overlap(self) -> None:
        diarization = FakeAnnotation([(FakeTurn(10.0, 12.0), "A", "SPEAKER_A")])

        assert _get_dominant_speaker(0.0, 5.0, diarization) == "SPEAKER_UNKNOWN"

    def test_accumulates_overlap_across_turns(self) -> None:
        diarization = FakeAnnotation(
            [
                (FakeTurn(0.0, 1.0), "A", "SPEAKER_A"),
                (FakeTurn(2.0, 3.0), "A2", "SPEAKER_A"),
                (FakeTurn(1.0, 2.5), "B", "SPEAKER_B"),
            ]
        )

        # SPEAKER_A: 1.0 + 1.0 = 2.0s, SPEAKER_B: 1.5s
        assert _get_dominant_speaker(0.0, 3.0, diarization) == "SPEAKER_A"
