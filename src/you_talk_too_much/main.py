from you_talk_too_much.app import AppSession
from you_talk_too_much.cli.logger import setup_logger
from you_talk_too_much.utils import get_key

logger = setup_logger(__name__)


def display_menu() -> None:
    """Display the application menu."""
    logger.info("Press the following key:")
    logger.info("1) Start new capture")
    logger.info("2) Stop existing capture")
    logger.info("3) Quit program")


def run() -> None:
    """Main application loop."""
    display_menu()

    session = AppSession()
    is_capture_started = False

    while True:
        key = get_key()
        if key == "1" and not is_capture_started:
            session.start()
            is_capture_started = True
        elif key == "2" and is_capture_started:
            session.stop()
            display_menu()
            is_capture_started = False
        elif key == "3":
            if is_capture_started:
                session.stop()
            logger.info("Quitting the program...")
            break

    logger.info("Done!")


if __name__ == "__main__":
    run()
