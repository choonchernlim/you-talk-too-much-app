import logging
from logging.handlers import RotatingFileHandler

from you_talk_too_much.cli.logger import setup_logger


class TestSetupLogger:
    def test_attaches_console_and_file_handlers(self) -> None:
        logger = setup_logger("test_logger_handlers")

        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types
        assert RotatingFileHandler in handler_types

    def test_no_duplicate_handlers_on_repeated_setup(self) -> None:
        logger_first = setup_logger("test_logger_duplicates")
        num_handlers = len(logger_first.handlers)

        logger_second = setup_logger("test_logger_duplicates")

        assert logger_second is logger_first
        assert len(logger_second.handlers) == num_handlers

    def test_file_handler_is_shared_across_loggers(self) -> None:
        logger_a = setup_logger("test_logger_shared_a")
        logger_b = setup_logger("test_logger_shared_b")

        file_handlers_a = [
            h for h in logger_a.handlers if isinstance(h, RotatingFileHandler)
        ]
        file_handlers_b = [
            h for h in logger_b.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert file_handlers_a[0] is file_handlers_b[0]
