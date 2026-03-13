import logging
import sys

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

        log_color = COLORS.get(record.levelno, RESET)
        message = super().format(record)

        record.name = original_name  # Restore original name
        return f"{log_color}{message}{RESET}"


def setup_logger(name: str) -> logging.Logger:
    """Setup logging with colored output."""
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

    return logger
