import numpy as np

from you_talk_too_much.audio.capturer import AudioCapturer
from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.config import settings
from you_talk_too_much.integrations.onenote import OneNoteClient
from you_talk_too_much.llm.summarizer import LLM
from you_talk_too_much.storage.file_manager import FileManager
from you_talk_too_much.transcription.transcriber import MLXTranscriber

logger = setup_logger(__name__)


class AppSession:
    """Manages the lifecycle of a capture session."""

    def __init__(self) -> None:
        """Initialize the application session and services."""
        self.file_manager = FileManager()
        self.transcriber = MLXTranscriber()
        self.audio_capturer = AudioCapturer(on_audio_ready=self._on_audio_ready)

        self.llm = LLM(
            settings.gcp_vertex_project,
            settings.gcp_vertex_location,
            settings.gcp_vertex_sa_key,
            settings.gcp_vertex_model,
        )
        self.onenote_client = OneNoteClient(
            settings.onenote_section_name,
            settings.azure_client_id,
            settings.azure_tenant_id,
        )

    def _on_audio_ready(self, audio_data: np.ndarray) -> None:
        """Callback invoked when audio chunk is ready for transcription."""
        raw_json_str, formatted_text = self.transcriber.process(audio_data)

        if raw_json_str:
            self.file_manager.append_raw_data(raw_json_str)

        if formatted_text:
            logger.info("\n" + formatted_text.strip())
            self.file_manager.append_conversation(formatted_text)

    def start(self) -> None:
        """Start a new capture session."""
        logger.info("Starting new capture...")
        self.file_manager.create_new_transcript_directory()
        self.transcriber.reset()
        self.audio_capturer.start()
        logger.info("Listening...")

    def tick(self) -> None:
        """Process accumulated audio if silence detected."""
        self.audio_capturer.tick()

    def stop(self) -> None:
        """Stop the current capture session and process the summary."""
        logger.info("Stopping existing capture...")
        self.audio_capturer.stop()

        # Post-processing (fail-fast)
        conversation_text = self.file_manager.read_conversation()
        if conversation_text.strip():
            markdown_summary, html_summary = self.llm.summarize(conversation_text)
            self.file_manager.write_summary(markdown_summary, html_summary)

            self.onenote_client.create_page(
                title=self.file_manager.get_formatted_datetime(),
                html_summary=html_summary,
            )

        logger.info("Stopped.")
