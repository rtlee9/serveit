"""Serve PyTorch ResNet50 model trained on ImageNet.

Prediction endpoint, served at `/predictions` takes a URL pointing to an image
and returns a list of class probabilities.
"""
from serveit.server import ModelServer
from serveit.utils import get_bytes_to_image_callback

import torchvision.models as models
import torchvision.transforms as transforms
import torch

from flask import request
import requests

# URL for ImageNet labels in JSON
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# parse labels into lookup
labels = {
    int(key): value for (key, value)
    in requests.get(LABELS_URL).json().items()
}

# load Resnet50 model pretrained on ImageNet
model = models.resnet50(pretrained=True)
model.eval()


# define a loader callback for the API to fetch the relevant data and
# convert to a format expected by the prediction function
def loader():
    """Load image from URL, and preprocess for Resnet."""
    url = request.args.get('url')  # read image URL as a request URL param
    response = requests.get(url)  # make request to static image file
    return response.content

#  define preprocessing callback chain
preprocessor = [
    get_bytes_to_image_callback(image_dims=(224, 224)),  # convert bytes to image of size 224 x 224
    lambda img: torch.from_numpy(img.swapaxes(3, 1).swapaxes(2, 3).copy()) / 255,  # convert to tensor, rescale
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize pixel intensities
    torch.autograd.Variable,  # convert to PyTorch Variable
]


# define a postprocessor callback for the API to transform the model predictions
def postprocessor(prediction):
    """Map prediction tensor to labels."""
    prediction = prediction.data.numpy()[0]
    top_predictions = prediction.argsort()[-3:][::-1]
    return [labels[prediction] for prediction in top_predictions]

# deploy model to a ModelServer
server = ModelServer(
    model,
    model,
    data_loader=loader,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    to_numpy=False
)

# start API
server.serve()
