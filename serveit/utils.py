"""Utility methods."""
import json
from .log_utils import get_logger

logger = get_logger(__name__)


def make_serializable(data):
    """Ensure data is serializable."""
    if is_serializable(data):
        return data

    # if numpy array convert to list
    try:
        return data.tolist()
    except AttributeError:
        pass

    # try serializing each child element
    if isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}
    if hasattr(data, '__iter__'):
        try:
            return [make_serializable(element) for element in data]
        except Exception:
            logger.debug('Could not serialize {}; converting to string'.format(data))

    # last resort: convert to string
    return str(data)


def is_serializable(data):
    """Check if data is serializable."""
    try:
        json.dumps(data)
        return True
    except TypeError:
        return False
