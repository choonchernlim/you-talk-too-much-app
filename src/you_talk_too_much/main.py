import time
import warnings

from you_talk_too_much.app import AppSession
from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.cli.terminal import poll_key

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

logger = setup_logger(__name__)

TICK_INTERVAL = 2.0


def display_menu() -> None:
    """Display the application menu."""
    logger.info("Press the following key:")
    logger.info("1) Start new capture")
    logger.info("2) Stop existing capture")
    logger.info("3) Summarize conversation")
    logger.info("4) Quit program")


def run() -> None:
    """Main application loop."""
    session = AppSession()
    is_capture_started = False
    last_tick = 0.0

    display_menu()

    while True:
        key = poll_key(timeout=0.1)

        if key == "1" and not is_capture_started:
            try:
                session.start()
                is_capture_started = True
                last_tick = time.monotonic()
            except Exception:
                logger.exception("Failed to start capture.")
                display_menu()
        elif key == "2" and is_capture_started:
            try:
                session.stop()
            except Exception:
                logger.exception("Error while stopping capture.")
            display_menu()
            is_capture_started = False
        elif key == "3" and not is_capture_started:
            prompt = "Enter transcript directory name (e.g. 2026-04-09 AM 08:08): "
            dir_name = input(prompt).strip()
            try:
                session.summarize_existing(dir_name)
            except Exception:
                logger.exception("Failed to summarize existing transcript.")
            display_menu()
        elif key == "4":
            if is_capture_started:
                try:
                    session.stop()
                except Exception:
                    logger.exception("Error while stopping capture.")
            logger.info("Quitting the program...")
            break

        if is_capture_started:
            now = time.monotonic()
            if now - last_tick >= TICK_INTERVAL:
                try:
                    session.tick()
                except Exception:
                    logger.exception(
                        "Error while processing audio chunk. "
                        "Skipping it and continuing to record."
                    )
                last_tick = now

    logger.info("Done!")


if __name__ == "__main__":
    run()
