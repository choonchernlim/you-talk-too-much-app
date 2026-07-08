import time
import warnings
from dataclasses import dataclass

from you_talk_too_much.app import AppSession
from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.cli.terminal import poll_key

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

logger = setup_logger(__name__)

TICK_INTERVAL = 2.0


@dataclass
class LoopState:
    """Mutable state of the main menu loop."""

    is_capture_started: bool = False
    last_tick: float = 0.0
    should_quit: bool = False


def display_menu() -> None:
    """Display the application menu."""
    logger.info("Press the following key:")
    logger.info("1) Start new capture")
    logger.info("2) Stop existing capture")
    logger.info("3) Summarize conversation")
    logger.info("4) Quit program")


def handle_key(session: AppSession, state: LoopState, key: str) -> None:
    """Handle a single menu keypress, updating the loop state."""
    if key == "1" and not state.is_capture_started:
        try:
            session.start()
            state.is_capture_started = True
            state.last_tick = time.monotonic()
        except Exception:
            logger.exception("Failed to start capture.")
            display_menu()
    elif key == "2" and state.is_capture_started:
        try:
            session.stop()
        except Exception:
            logger.exception("Error while stopping capture.")
        display_menu()
        state.is_capture_started = False
    elif key == "3" and not state.is_capture_started:
        prompt = "Enter transcript directory name (e.g. 2026-04-09 AM 08:08): "
        dir_name = input(prompt).strip()
        try:
            session.summarize_existing(dir_name)
        except Exception:
            logger.exception("Failed to summarize existing transcript.")
        display_menu()
    elif key == "4":
        state.should_quit = True


def handle_tick(session: AppSession, state: LoopState) -> None:
    """Run the periodic VAD tick if a capture is active and due."""
    if not state.is_capture_started:
        return
    now = time.monotonic()
    if now - state.last_tick >= TICK_INTERVAL:
        try:
            session.tick()
        except Exception:
            logger.exception(
                "Error while processing audio chunk. "
                "Skipping it and continuing to record."
            )
        state.last_tick = now


def run() -> None:
    """Main application loop."""
    session = AppSession()
    state = LoopState()

    display_menu()

    try:
        while not state.should_quit:
            key = poll_key(timeout=0.1)
            if key:
                handle_key(session, state, key)
            handle_tick(session, state)
    finally:
        # Reached via quit (4), Ctrl-C, or an unexpected error: stop any
        # active capture and wait for pending transcription/summary work.
        logger.info("Quitting the program...")
        session.shutdown()

    logger.info("Done!")


if __name__ == "__main__":
    run()
