from threading import Event, Thread

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

        self.stop_event = Event()
        self.capture_thread: Thread | None = None
        self.process_thread: Thread | None = None

    def _on_audio_ready(self, audio_data: np.ndarray) -> None:
        """Callback invoked when audio chunk is ready for transcription."""
        raw_json_str, formatted_text = self.transcriber.process(audio_data)

        if raw_json_str:
            self.file_manager.append_raw_data(raw_json_str)

        if formatted_text:
            logger.info(f"Conversation:\n{formatted_text.strip()}")
            self.file_manager.append_conversation(formatted_text)

    def start(self) -> None:
        """Start the audio capture and processing threads."""
        logger.info("Starting new capture...")
        self.file_manager.create_new_transcript_directory()
        self.stop_event.clear()

        self.capture_thread = Thread(
            target=self.audio_capturer.capture_audio, args=(self.stop_event,), daemon=True
        )
        self.process_thread = Thread(
            target=self.audio_capturer.batch_process_buffer,
            args=(self.stop_event,),
            daemon=True,
        )

        self.capture_thread.start()
        self.process_thread.start()
        logger.info("Listening...")

    def stop(self) -> None:
        """Stop the current capture session and process the summary."""
        logger.info("Stopping existing capture...")
        self.stop_event.set()

        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()

        # Post-processing
        try:
            conversation_text = self.file_manager.read_conversation()
            if conversation_text.strip():
                markdown_summary, html_summary = self.llm.summarize(conversation_text)
                if markdown_summary and html_summary:
                    self.file_manager.write_summary(markdown_summary, html_summary)

                    self.onenote_client.create_page(
                        title=self.file_manager.get_formatted_datetime(),
                        html_summary=html_summary,
                    )
        except Exception as e:
            logger.error(f"Error during post-processing: {e}")

        logger.info("Stopped.")
