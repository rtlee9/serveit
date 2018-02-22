"""Logger setup."""
import logging
from os import getenv

logging.basicConfig(level=logging.WARNING)


def get_logger(name):
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(getenv('LOGLEVEL', 'INFO'))
    return logger
