from pathlib import Path

import pytest

from you_talk_too_much.storage.file_manager import FileManager, TranscriptSession


class TestFileManagerCreateNew:
    def test_creates_directory_and_returns_session(self, tmp_path: Path) -> None:
        fm = FileManager(base_dir=str(tmp_path))

        session = fm.create_new()

        assert session.out_dir.exists()
        assert session.out_dir.parent == tmp_path
        assert session.formatted_datetime == session.out_dir.name


class TestFileManagerLoadExisting:
    def test_returns_session_for_existing_directory(self, tmp_path: Path) -> None:
        (tmp_path / "2026-04-09 AM 08:08").mkdir()
        fm = FileManager(base_dir=str(tmp_path))

        session = fm.load_existing("2026-04-09 AM 08:08")

        assert session.out_dir == tmp_path / "2026-04-09 AM 08:08"
        assert session.formatted_datetime == "2026-04-09 AM 08:08"

    def test_raises_for_missing_directory(self, tmp_path: Path) -> None:
        fm = FileManager(base_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError):
            fm.load_existing("nope")


class TestTranscriptSession:
    def _session(self, tmp_path: Path) -> TranscriptSession:
        return TranscriptSession(out_dir=tmp_path, formatted_datetime="x")

    def test_read_conversation_returns_empty_when_missing(self, tmp_path: Path) -> None:
        assert self._session(tmp_path).read_conversation() == ""

    def test_append_and_read_conversation(self, tmp_path: Path) -> None:
        session = self._session(tmp_path)

        session.append_conversation("SPEAKER_00: Hello\n")
        session.append_conversation("SPEAKER_01: Hi\n")

        assert session.read_conversation() == "SPEAKER_00: Hello\nSPEAKER_01: Hi\n"

    def test_append_raw_data(self, tmp_path: Path) -> None:
        session = self._session(tmp_path)

        session.append_raw_data('{"a": 1}\n')

        assert (tmp_path / "raw.jsonl").read_text() == '{"a": 1}\n'

    def test_write_summary_creates_md_and_html(self, tmp_path: Path) -> None:
        session = self._session(tmp_path)

        session.write_summary("# md", "<h1>html</h1>")

        assert (tmp_path / "conversation.md").read_text() == "# md"
        assert (tmp_path / "conversation.html").read_text() == "<h1>html</h1>"

    def test_is_immutable(self, tmp_path: Path) -> None:
        session = self._session(tmp_path)

        with pytest.raises(AttributeError):
            session.out_dir = tmp_path / "other"  # type: ignore[misc]
