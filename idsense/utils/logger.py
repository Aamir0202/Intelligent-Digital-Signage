import logging
import sys

from .config import Config


class Logger:
    _inner = None

    LOG_LEVEL_ALWAYS = 9999

    @staticmethod
    def init():
        """Initializes the logger with file and console handlers."""

        logging.addLevelName(Logger.LOG_LEVEL_ALWAYS, "VERBOSE")

        logger = logging.getLogger(name="idsense")
        logger.propagate = False

        file_handler = logging.FileHandler(Config.get("logFile"))
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("[%(asctime)s | %(levelname)s]: %(message)s")
        file_handler.setFormatter(file_formatter)

        numeric_level = getattr(
            logger, "DEBUG" if Config.get("debug") else "WARNING", logging.INFO
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            "[%(levelname)s | %(asctime)s]: %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        Logger._inner = logger

    @staticmethod
    def log(msg):
        """Logs a message with 'VERBOSE' level."""

        return Logger._inner.log(Logger.LOG_LEVEL_ALWAYS, msg)

    @staticmethod
    def info(msg, *args):
        """Logs an 'INFO' level message."""

        return Logger._inner.info(msg, *args)

    @staticmethod
    def warning(msg, *args):
        """Logs a 'WARNING' level message."""

        return Logger._inner.warning(msg, *args)

    @staticmethod
    def error(msg, *args):
        """Logs an 'ERROR' level message."""

        return Logger._inner.error(msg, *args)
