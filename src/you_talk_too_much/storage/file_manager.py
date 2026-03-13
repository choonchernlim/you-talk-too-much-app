import os
from datetime import datetime

from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.utils import append_file, read_file, write_file

logger = setup_logger(__name__)


class FileManager:
    """Manages file storage for transcripts and summaries."""

    def __init__(self, base_dir: str = "transcripts") -> None:
        """Initialize the file manager."""
        self.base_dir = base_dir
        self.out_dir = ""
        self.formatted_datetime = ""

    def create_new_transcript_directory(self) -> None:
        """Create a new directory for storing transcripts."""
        logger.info("Creating new transcript directory...")
        self.formatted_datetime = datetime.now().strftime("%Y-%m-%d %p %I:%M")
        self.out_dir = os.path.join(self.base_dir, self.formatted_datetime)
        os.makedirs(self.out_dir, exist_ok=True)

    def get_formatted_datetime(self) -> str:
        """Return the formatted datetime of the current session."""
        return self.formatted_datetime

    def get_conversation_file_path(self) -> str:
        """Return the path to the conversation text file."""
        if not self.out_dir:
            raise ValueError("Directory not created yet.")
        return os.path.join(self.out_dir, "conversation.txt")

    def get_raw_file_path(self) -> str:
        """Return the path to the raw transcription JSONL file."""
        if not self.out_dir:
            raise ValueError("Directory not created yet.")
        return os.path.join(self.out_dir, "raw.jsonl")

    def append_raw_data(self, data: str) -> None:
        """Append data to the raw JSONL file."""
        append_file(self.get_raw_file_path(), data)

    def append_conversation(self, data: str) -> None:
        """Append data to the conversation text file."""
        append_file(self.get_conversation_file_path(), data)

    def write_summary(self, markdown_content: str, html_content: str) -> None:
        """Write the summary to markdown and html files."""
        if not self.out_dir:
            raise ValueError("Directory not created yet.")
        base_path = os.path.join(self.out_dir, "conversation")
        write_file(f"{base_path}.md", markdown_content)
        write_file(f"{base_path}.html", html_content)

    def read_conversation(self) -> str:
        """Read the conversation text file."""
        return read_file(self.get_conversation_file_path())
