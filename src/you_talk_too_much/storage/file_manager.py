from datetime import datetime
from pathlib import Path

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)


class FileManager:
    """Manages file storage for transcripts and summaries."""

    def __init__(self, base_dir: str = "transcripts") -> None:
        """Initialize the file manager."""
        self.base_dir = Path(base_dir)
        self.out_dir: Path | None = None
        self.formatted_datetime = ""

    def create_new_transcript_directory(self) -> None:
        """Create a new directory for storing transcripts."""
        logger.info("Creating new transcript directory...")
        self.formatted_datetime = datetime.now().strftime("%Y-%m-%d %p %I:%M")
        self.out_dir = self.base_dir / self.formatted_datetime
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_existing_transcript_directory(self, dir_name: str) -> None:
        """Load an existing transcript directory by name."""
        self.out_dir = self.base_dir / dir_name
        if not self.out_dir.exists():
            raise FileNotFoundError(f"Transcript directory not found: {self.out_dir}")
        self.formatted_datetime = dir_name

    def get_formatted_datetime(self) -> str:
        """Return the formatted datetime of the current session."""
        return self.formatted_datetime

    def get_conversation_file_path(self) -> str:
        """Return the path to the conversation text file."""
        if not self.out_dir:
            raise ValueError("Directory not created yet.")
        return str(self.out_dir / "conversation.txt")

    def get_raw_file_path(self) -> str:
        """Return the path to the raw transcription JSONL file."""
        if not self.out_dir:
            raise ValueError("Directory not created yet.")
        return str(self.out_dir / "raw.jsonl")

    def append_raw_data(self, data: str) -> None:
        """Append data to the raw JSONL file."""
        with Path(self.get_raw_file_path()).open("a") as f:
            f.write(data)

    def append_conversation(self, data: str) -> None:
        """Append data to the conversation text file."""
        with Path(self.get_conversation_file_path()).open("a") as f:
            f.write(data)

    def write_summary(self, markdown_content: str, html_content: str) -> None:
        """Write the summary to markdown and html files."""
        if not self.out_dir:
            raise ValueError("Directory not created yet.")
        base_path = self.out_dir / "conversation"
        Path(f"{base_path}.md").write_text(markdown_content)
        Path(f"{base_path}.html").write_text(html_content)

    def read_conversation(self) -> str:
        """Read the conversation text file, returning empty string if not found."""
        path = Path(self.get_conversation_file_path())
        if not path.exists():
            return ""
        return path.read_text()
