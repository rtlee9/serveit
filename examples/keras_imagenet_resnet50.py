"""Serve Keras ResNet50 model trained on ImageNet.

Prediction endpoint, served at `/predictions` takes a URL pointing to an image
and returns a list of class probabilities.
"""
from serveit.server import ModelServer
from serveit.utils import make_serializable, get_bytes_to_image_callback

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import preprocess_input

from flask import request
import requests

# load Resnet50 model pretrained on ImageNet
model = ResNet50(weights='imagenet')


# define a loader callback for the API to fetch the relevant data and
# convert to a format expected by the prediction function
def loader():
    """Load image from URL, and preprocess for Resnet."""
    url = request.args.get('url')  # read image URL as a request URL param
    response = requests.get(url)  # make request to static image file
    return response.content

# get a bytes-to-image callback, resizing the image to 224x224 for ImageNet
bytes_to_image = get_bytes_to_image_callback(image_dims=(224, 224))


# define a postprocessor callback for the API to transform the model predictions
def postprocessor(predictions):
    """Decode predictions and serialize."""
    # decode all class predictions and take top 3
    top_predictions = decode_predictions(predictions, top=3)[0]
    # serialize predictions for JSON response
    return make_serializable(top_predictions)

# deploy model to a ModelServer
server = ModelServer(
    model,
    model.predict,
    data_loader=loader,
    preprocessor=[bytes_to_image, preprocess_input],
    postprocessor=postprocessor,
)

# start API
server.serve()
