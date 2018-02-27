"""Serve Keras ResNet50 model trained on ImageNet.

Prediction endpoint, served at `/predictions` takes a URL pointing to an image
and returns a list of class probabilities.
"""
from serveit.server import ModelServer
from serveit.utils import make_serializable

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from flask import request
from PIL import Image
import requests
from io import BytesIO

# load Resnet50 model pretrained on ImageNet
model = ResNet50(weights='imagenet')


# define a loader callback for the API to fetch the relevant data and
# convert to a format expected by the prediction function
def loader():
    """Load image from URL, and preprocess for Resnet."""
    url = request.args.get('url')  # read image URL as a request URL param
    response = requests.get(url)  # make request to static image file
    img = Image.open(BytesIO(response.content))  # open image
    img = img.resize((224, 224), Image.ANTIALIAS)  # model requires 224x224 pixels
    x = image.img_to_array(img)  # convert image to numpy array
    x = np.expand_dims(x, axis=0)  # model expects dim 0 to be iterable across images
    return preprocess_input(x)  # preprocess the image using keras fn


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
    postprocessor=postprocessor,
)

# start API
server.serve()
