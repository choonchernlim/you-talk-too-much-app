from pathlib import Path

from you_talk_too_much.storage.file_manager import FileManager


class TestFileManagerReadConversation:
    def test_returns_empty_string_when_file_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        fm = FileManager(base_dir=str(tmp_path))
        fm.out_dir = tmp_path / "nonexistent_session"

        result = fm.read_conversation()

        assert result == ""

    def test_returns_content_when_file_exists(self, tmp_path: Path) -> None:
        fm = FileManager(base_dir=str(tmp_path))
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        fm.out_dir = session_dir

        conv_file = session_dir / "conversation.txt"
        conv_file.write_text("SPEAKER_00: Hello world\n")

        result = fm.read_conversation()

        assert result == "SPEAKER_00: Hello world\n"
