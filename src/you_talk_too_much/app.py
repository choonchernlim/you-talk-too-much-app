import numpy as np
from markdown import markdown

from you_talk_too_much.audio.capturer import AudioCapturer
from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.common.task_runner import TaskRunner
from you_talk_too_much.config import get_settings
from you_talk_too_much.integrations.onenote import OneNoteClient
from you_talk_too_much.llm.summarizer import LLM
from you_talk_too_much.storage.file_manager import FileManager, TranscriptSession
from you_talk_too_much.transcription.transcriber import MLXTranscriber

logger = setup_logger(__name__)


class AppSession:
    """Manages the lifecycle of capture sessions.

    Thread model (see plan): the main thread only polls keys and runs the
    cheap VAD tick. All heavy work runs on two serial lanes:
    - `transcription` runner: whisper/diarization + transcript file writes
    - `summarize` runner: LLM + OneNote calls
    Jobs close over an immutable TranscriptSession, so a new recording can
    start while the previous session's work is still in flight.
    """

    def __init__(self) -> None:
        """Initialize the application session and services."""
        settings = get_settings()
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

        self.transcription = TaskRunner("transcription")
        self.summarize = TaskRunner("summarize")
        self.current_session: TranscriptSession | None = None

        # Load the lazily-initialized whisper model before the first real
        # chunk needs it; runs in the background while the menu is shown.
        self.transcription.submit(self._warm_up_whisper)

    def _warm_up_whisper(self) -> None:
        """Transcribe one second of silence to populate the model cache."""
        logger.info("Warming up transcription model...")
        self.transcriber.transcribe(np.zeros(16000, dtype=np.float32))
        logger.info("Transcription model ready.")

    def _on_audio_ready(self, audio_data: np.ndarray) -> None:
        """Callback invoked when an audio chunk is ready for transcription."""
        session = self.current_session
        if session is None:
            logger.warning("Audio chunk received with no active session; dropping.")
            return
        self.transcription.submit(lambda: self._process_chunk(audio_data, session))

    def _process_chunk(
        self, audio_data: np.ndarray, session: TranscriptSession
    ) -> None:
        """Transcribe one chunk and append it to the session's transcript."""
        raw_json_str, formatted_text = self.transcriber.process(audio_data)

        if raw_json_str:
            session.append_raw_data(raw_json_str)

        if formatted_text:
            logger.info("\n" + formatted_text.strip())
            session.append_conversation(formatted_text)

    def start(self) -> None:
        """Start a new capture session."""
        logger.info("Starting new capture...")
        # Reset runs on the transcription lane so it is ordered after any
        # still-pending chunks of the previous session.
        self.transcription.submit(self.transcriber.reset)
        self.current_session = self.file_manager.create_new()
        self.audio_capturer.start()
        logger.info("Listening...")

    def tick(self) -> None:
        """Process accumulated audio if silence detected."""
        self.audio_capturer.tick()

    def stop(self) -> None:
        """Stop the capture; transcription and summary finish in the background."""
        logger.info("Stopping existing capture...")
        try:
            self.audio_capturer.stop()
        finally:
            session = self.current_session
            self.current_session = None
            if session is not None:
                # Runs after every chunk of this session has been
                # transcribed, then hands off to the summarize lane.
                self.transcription.submit(
                    lambda: self.summarize.submit(
                        lambda: self._summarize_and_push(session)
                    )
                )
                logger.info(
                    f"Recording stopped. Summary for '{session.formatted_datetime}' "
                    "will be generated and pushed to OneNote in the background."
                )

    def _summarize_and_push(self, session: TranscriptSession) -> None:
        """Summarize a session's conversation and push it to OneNote."""
        conversation_text = session.read_conversation()
        if not conversation_text.strip():
            logger.info("No conversation text found. Nothing to summarize.")
            return

        try:
            markdown_summary, topic = self.llm.summarize(conversation_text)
            html_summary = markdown(markdown_summary)
            session.write_summary(markdown_summary, html_summary)

            self.onenote_client.create_page(
                title=f"{session.formatted_datetime} - WHO - {topic}",
                html_summary=html_summary,
            )
            logger.info(
                f"Summary for '{session.formatted_datetime}' pushed to OneNote."
            )
        except Exception:
            logger.exception(
                "Summarization/OneNote push failed. The transcript is saved; "
                f"use menu option 3 with directory name "
                f"'{session.formatted_datetime}' to retry."
            )

    def summarize_existing(self, dir_name: str) -> None:
        """Run summarization and OneNote push for an existing transcript directory."""
        assert "/" not in dir_name, f"dir_name must not contain '/': {dir_name!r}"
        assert not dir_name.startswith("."), (
            f"dir_name must not start with '.': {dir_name!r}"
        )

        session = self.file_manager.load_existing(dir_name)
        logger.info(f"Summarizing existing transcript: {dir_name}")
        # Route through the summarize lane so the LLM/OneNote clients are
        # only ever used by one thread, then wait for the result.
        self.summarize.submit(lambda: self._summarize_and_push(session))
        self.summarize.join()
        logger.info("Done.")

    def shutdown(self) -> None:
        """Stop any active capture and wait for all background work to finish."""
        if self.current_session is not None:
            self.stop()
        logger.info("Waiting for pending transcription/summary work...")
        self.transcription.shutdown()
        self.summarize.shutdown()
        logger.info("All pending work complete.")
