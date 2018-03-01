"""Utility methods."""
import json
from flask import request

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


def json_numpy_loader():
    """Load data from JSON request and convert to numpy array."""
    data = request.get_json()
    logger.debug('Received JSON data of length {:,}'.format(len(data)))
    return data


def get_bytes_to_image_callback(image_dims=(224, 224)):
    """Return a callback to process image bytes for ImageNet."""
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input
    import numpy as np
    from PIL import Image
    from io import BytesIO

    def preprocess_image_bytes(data_bytes):
        """Process image bytes for ImageNet."""
        img = Image.open(BytesIO(data_bytes))  # open image
        img = img.resize(image_dims, Image.ANTIALIAS)  # model requires 224x224 pixels
        x = image.img_to_array(img)  # convert image to numpy array
        x = np.expand_dims(x, axis=0)  # model expects dim 0 to be iterable across images
        return preprocess_input(x)  # preprocess the image using keras fn
    return preprocess_image_bytes
