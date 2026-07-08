import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from you_talk_too_much.app import AppSession
from you_talk_too_much.storage.file_manager import TranscriptSession


@patch("you_talk_too_much.app.OneNoteClient")
@patch("you_talk_too_much.app.LLM")
@patch("you_talk_too_much.app.AudioCapturer")
@patch("you_talk_too_much.app.MLXTranscriber")
@patch("you_talk_too_much.app.FileManager")
class TestAppSession:
    def _make_session(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        mock_llm: MagicMock,
        tmp_path: Path,
        dir_names: list[str],
    ) -> AppSession:
        """Build an AppSession whose create_new() yields real session dirs."""
        transcript_sessions = []
        for name in dir_names:
            out_dir = tmp_path / name
            out_dir.mkdir()
            transcript_sessions.append(
                TranscriptSession(out_dir=out_dir, formatted_datetime=name)
            )
        mock_file_manager.return_value.create_new.side_effect = transcript_sessions
        mock_transcriber.return_value.process.return_value = (
            '{"raw": 1}\n',
            "SPEAKER_00: hi\n",
        )
        mock_llm.return_value.summarize.return_value = ("# summary", "topic")
        return AppSession()

    def test_stop_summarizes_transcribed_chunks_in_background(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        mock_capturer: MagicMock,
        mock_llm: MagicMock,
        mock_onenote: MagicMock,
        tmp_path: Path,
    ) -> None:
        session = self._make_session(
            mock_file_manager, mock_transcriber, mock_llm, tmp_path, ["s1"]
        )

        session.start()
        session._on_audio_ready(np.zeros(10, dtype=np.float32))
        session.stop()
        session.shutdown()

        assert (tmp_path / "s1" / "conversation.txt").read_text() == "SPEAKER_00: hi\n"
        mock_capturer.return_value.stop.assert_called_once()
        mock_llm.return_value.summarize.assert_called_once_with("SPEAKER_00: hi\n")
        mock_onenote.return_value.create_page.assert_called_once()
        title = mock_onenote.return_value.create_page.call_args.kwargs["title"]
        assert title.startswith("s1 - ")

    def test_new_recording_can_start_while_previous_summary_in_flight(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        _mock_capturer: MagicMock,
        mock_llm: MagicMock,
        mock_onenote: MagicMock,
        tmp_path: Path,
    ) -> None:
        session = self._make_session(
            mock_file_manager, mock_transcriber, mock_llm, tmp_path, ["s1", "s2"]
        )
        gate = threading.Event()

        def slow_summarize(_text: str) -> tuple[str, str]:
            assert gate.wait(timeout=5)
            return ("# summary", "topic")

        mock_llm.return_value.summarize.side_effect = slow_summarize

        # First session: stop kicks off a summarize that blocks on the gate
        session.start()
        session._on_audio_ready(np.zeros(10, dtype=np.float32))
        session.stop()

        # Second session starts and transcribes while the first summary hangs
        mock_transcriber.return_value.process.return_value = (
            '{"raw": 2}\n',
            "SPEAKER_01: bye\n",
        )
        session.start()
        session._on_audio_ready(np.zeros(10, dtype=np.float32))
        session.transcription.join()

        # Second transcript is already on disk before the first summary is done
        assert (tmp_path / "s2" / "conversation.txt").read_text() == "SPEAKER_01: bye\n"
        assert not (tmp_path / "s1" / "conversation.md").exists()

        gate.set()
        session.stop()
        session.shutdown()

        # Both sessions summarized independently, in order
        summarized = [
            call.args[0] for call in mock_llm.return_value.summarize.call_args_list
        ]
        assert summarized == ["SPEAKER_00: hi\n", "SPEAKER_01: bye\n"]
        assert mock_onenote.return_value.create_page.call_count == 2
        assert (tmp_path / "s1" / "conversation.md").exists()
        assert (tmp_path / "s2" / "conversation.md").exists()

    def test_summarize_failure_is_contained(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        mock_capturer: MagicMock,
        mock_llm: MagicMock,
        mock_onenote: MagicMock,
        tmp_path: Path,
    ) -> None:
        session = self._make_session(
            mock_file_manager, mock_transcriber, mock_llm, tmp_path, ["s1"]
        )
        mock_llm.return_value.summarize.side_effect = RuntimeError("LLM down")

        session.start()
        session._on_audio_ready(np.zeros(10, dtype=np.float32))
        session.stop()
        session.shutdown()  # must not raise

        mock_capturer.return_value.stop.assert_called_once()
        mock_onenote.return_value.create_page.assert_not_called()
        # Transcript survives for a retry via menu option 3
        assert (tmp_path / "s1" / "conversation.txt").exists()

    def test_chunk_without_active_session_is_dropped(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        _mock_capturer: MagicMock,
        mock_llm: MagicMock,
        _mock_onenote: MagicMock,
        tmp_path: Path,
    ) -> None:
        session = self._make_session(
            mock_file_manager, mock_transcriber, mock_llm, tmp_path, []
        )

        session._on_audio_ready(np.zeros(10, dtype=np.float32))
        session.shutdown()

        mock_transcriber.return_value.process.assert_not_called()

    def test_warm_up_transcribes_silence_at_startup(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        _mock_capturer: MagicMock,
        mock_llm: MagicMock,
        _mock_onenote: MagicMock,
        tmp_path: Path,
    ) -> None:
        session = self._make_session(
            mock_file_manager, mock_transcriber, mock_llm, tmp_path, []
        )
        session.transcription.join()

        mock_transcriber.return_value.transcribe.assert_called_once()
        warm_up_audio = mock_transcriber.return_value.transcribe.call_args.args[0]
        assert warm_up_audio.shape == (16000,)
        session.shutdown()

    def test_summarize_existing_blocks_until_pushed(
        self,
        mock_file_manager: MagicMock,
        mock_transcriber: MagicMock,
        _mock_capturer: MagicMock,
        mock_llm: MagicMock,
        mock_onenote: MagicMock,
        tmp_path: Path,
    ) -> None:
        session = self._make_session(
            mock_file_manager, mock_transcriber, mock_llm, tmp_path, []
        )
        existing_dir = tmp_path / "old"
        existing_dir.mkdir()
        (existing_dir / "conversation.txt").write_text("SPEAKER_00: old talk\n")
        mock_file_manager.return_value.load_existing.return_value = TranscriptSession(
            out_dir=existing_dir, formatted_datetime="old"
        )

        session.summarize_existing("old")

        # Blocking call: push already happened when the call returned
        mock_onenote.return_value.create_page.assert_called_once()
        assert (existing_dir / "conversation.md").read_text() == "# summary"
        session.shutdown()
