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
        log_color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
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
        formatter = ColoredFormatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger
