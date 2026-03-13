import os
from dataclasses import dataclass
from threading import Event, Thread

from dotenv import load_dotenv

from .audio_capturer import AudioCapturer
from .llm import LLM
from .log_config import setup_logger
from .onenote_client import OneNoteClient
from .transcriber import MLXTranscriber
from .utils import get_key

load_dotenv()

logger = setup_logger(__name__)

GCP_VERTEX_PROJECT = os.getenv("GCP_VERTEX_PROJECT", "")
GCP_VERTEX_LOCATION = os.getenv("GCP_VERTEX_LOCATION", "")
GCP_VERTEX_SA_KEY = os.getenv("GCP_VERTEX_SA_KEY", "")
GCP_VERTEX_MODEL = os.getenv("GCP_VERTEX_MODEL", "")
ONENOTE_SECTION_NAME = os.getenv("ONENOTE_SECTION_NAME", "")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")

transcriber = MLXTranscriber()
audio_capturer = AudioCapturer(transcriber)
llm = LLM(GCP_VERTEX_PROJECT, GCP_VERTEX_LOCATION, GCP_VERTEX_SA_KEY, GCP_VERTEX_MODEL)
onenote_client = OneNoteClient(ONENOTE_SECTION_NAME, AZURE_CLIENT_ID, AZURE_TENANT_ID)


@dataclass
class CaptureSession:
    """Session object to hold capture threads and event."""

    stop_event: Event
    capture_thread: Thread
    process_thread: Thread


_session: CaptureSession | None = None


def start_capture() -> None:
    """Start the audio capture and processing threads."""
    global _session  # noqa: PLW0603

    if _session is not None:
        return

    logger.info("Starting new capture...")

    stop_event = Event()
    capture_thread = Thread(
        target=audio_capturer.capture_audio, args=(stop_event,), daemon=True
    )
    process_thread = Thread(
        target=audio_capturer.batch_process_buffer, args=(stop_event,), daemon=True
    )

    _session = CaptureSession(
        stop_event=stop_event,
        capture_thread=capture_thread,
        process_thread=process_thread,
    )

    # Start audio capture in background threads
    capture_thread.start()
    process_thread.start()

    logger.info("Listening...")


def stop_capture() -> None:
    """Stop the current capture session and create OneNote page."""
    global _session  # noqa: PLW0603

    if _session is None:
        return

    logger.info("Stopping existing capture...")
    _session.stop_event.set()

    _session.capture_thread.join()

    _session.process_thread.join()

    # Only create OneNote page if conversation file exists
    conv_file = transcriber.get_conversation_file_path()
    if os.path.exists(conv_file):
        onenote_client.create_page(
            title=transcriber.get_formatted_datetime(),
            html_summary=llm.summarize(conv_file),
        )

    _session = None
    logger.info("Stopped.")


def display_menu() -> None:
    """Display the application menu."""
    logger.info("Press the following key:")
    logger.info("1) Start new capture")
    logger.info("2) Stop existing capture")
    logger.info("3) Quit program")


def run() -> None:
    """Main application loop."""
    display_menu()

    is_capture_started = False

    while True:
        key = get_key()
        if key == "1" and not is_capture_started:
            start_capture()
            is_capture_started = True
        elif key == "2" and is_capture_started:
            stop_capture()
            display_menu()
            is_capture_started = False
        elif key == "3":
            stop_capture()
            logger.info("Quitting the program...")
            break

    logger.info("Done!")


if __name__ == "__main__":
    run()
