import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FILE = Path.home() / ".you-talk-too-much" / "logs" / "app.log"
_MAX_LOG_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 5

# Mapping log levels to colors
COLORS: dict[int, str] = {
    logging.DEBUG: "\033[0;37m",  # White
    logging.INFO: "\033[0;32m",  # Green
    logging.WARNING: "\033[0;33m",  # Yellow
    logging.ERROR: "\033[0;31m",  # Red
    logging.CRITICAL: "\033[1;41m",  # Red background
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter to provide colored output."""

    def format(self, record: logging.LogRecord) -> str:
        """Apply color based on the level of the log message."""
        original_name = record.name
        parts = original_name.split(".")

        # Abbreviate intermediate parts to their first letter
        # e.g. "you_talk_too_much.transcription.transcriber" -> "y.t.transcriber"
        if len(parts) > 1:
            abbreviated_parts = [p[0] for p in parts[:-1]] + [parts[-1]]
            record.name = ".".join(abbreviated_parts)

        message = super().format(record)

        log_color = COLORS.get(record.levelno, RESET)
        record.name = original_name  # Restore original name
        return f"{log_color}{message}{RESET}"


_file_handler: RotatingFileHandler | None = None


def _get_file_handler() -> RotatingFileHandler:
    """Return the shared file handler, creating it on first use."""
    global _file_handler  # noqa: PLW0603
    if _file_handler is None:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = RotatingFileHandler(
            _LOG_FILE, maxBytes=_MAX_LOG_BYTES, backupCount=_LOG_BACKUP_COUNT
        )
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s [%(threadName)s] %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    return _file_handler


def setup_logger(name: str) -> logging.Logger:
    """Setup logging with colored console output and a persistent log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = ColoredFormatter(
            "%(asctime)s %(levelname)-5s %(name)-20s %(message)s",
            datefmt="%I:%M:%S%p",
        )
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)
        logger.addHandler(_get_file_handler())

    return logger
