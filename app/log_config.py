import logging


# Define ANSI color codes
class ColoredFormatter(logging.Formatter):
    # Mapping log levels to colors
    COLORS = {
        logging.DEBUG: "\033[0;37m",  # White
        logging.INFO: "\033[0;32m",  # Green
        logging.WARNING: "\033[0;33m",  # Yellow
        logging.ERROR: "\033[0;31m",  # Red
        logging.CRITICAL: "\033[1;41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        # Apply color based on the level of the log message
        log_color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"


# Setup logging with colored output
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a console handler
    ch = logging.StreamHandler()

    # Create a colored formatter
    formatter = ColoredFormatter('%(asctime)s %(levelname)-8s %(name)-20s %(message)s')

    # Set the formatter to the handler
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger
