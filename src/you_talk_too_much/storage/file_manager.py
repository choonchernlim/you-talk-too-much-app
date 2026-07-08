from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from you_talk_too_much.cli.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class TranscriptSession:
    """Immutable handle to one recording session's transcript directory.

    Instances are shared across threads (transcription and summarize
    runners); being frozen guarantees there is no mutable state to race on.
    """

    out_dir: Path
    formatted_datetime: str

    @property
    def conversation_path(self) -> Path:
        """Path to the conversation text file."""
        return self.out_dir / "conversation.txt"

    @property
    def raw_path(self) -> Path:
        """Path to the raw transcription JSONL file."""
        return self.out_dir / "raw.jsonl"

    def append_raw_data(self, data: str) -> None:
        """Append data to the raw JSONL file."""
        with self.raw_path.open("a") as f:
            f.write(data)

    def append_conversation(self, data: str) -> None:
        """Append data to the conversation text file."""
        with self.conversation_path.open("a") as f:
            f.write(data)

    def read_conversation(self) -> str:
        """Read the conversation text file, returning empty string if not found."""
        if not self.conversation_path.exists():
            return ""
        return self.conversation_path.read_text()

    def write_summary(self, markdown_content: str, html_content: str) -> None:
        """Write the summary to markdown and html files."""
        base_path = self.out_dir / "conversation"
        Path(f"{base_path}.md").write_text(markdown_content)
        Path(f"{base_path}.html").write_text(html_content)


class FileManager:
    """Creates and loads transcript session directories."""

    def __init__(self, base_dir: str = "transcripts") -> None:
        """Initialize the file manager."""
        self.base_dir = Path(base_dir)

    def create_new(self) -> TranscriptSession:
        """Create a new timestamped directory and return its session handle."""
        logger.info("Creating new transcript directory...")
        formatted_datetime = datetime.now().strftime("%Y-%m-%d %p %I:%M")
        out_dir = self.base_dir / formatted_datetime
        out_dir.mkdir(parents=True, exist_ok=True)
        return TranscriptSession(out_dir=out_dir, formatted_datetime=formatted_datetime)

    def load_existing(self, dir_name: str) -> TranscriptSession:
        """Load an existing transcript directory by name."""
        out_dir = self.base_dir / dir_name
        if not out_dir.exists():
            raise FileNotFoundError(f"Transcript directory not found: {out_dir}")
        return TranscriptSession(out_dir=out_dir, formatted_datetime=dir_name)
