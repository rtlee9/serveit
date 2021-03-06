# ServeIt examples

## Basic example: Iris predictions with Scikit-learn

Let's train and deploy a logistic regression model to classify irises. We'll start by fitting a model:
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# fit a model on the Iris dataset
data = load_iris()
clf = LogisticRegression()
clf.fit(data.data, data.target)
```
Now we can serve our trained model:
```python
from serveit.server import ModelServer

# initialize server
server = ModelServer(clf, clf.predict)

# optional: add informational endpoints
server.create_info_endpoint('features', data.feature_names)
server.create_info_endpoint('target_labels', data.target_names.tolist())

# start serving predictions from API
server.serve()
```

Behold:

```bash
curl -XPOST 'localhost:5000/predictions'\
    -H "Content-Type: application/json"\
    -d "[[5.6, 2.9, 3.6, 1.3], [4.4, 2.9, 1.4, 0.2], [5.5, 2.4, 3.8, 1.1], [5.0, 3.4, 1.5, 0.2], [5.7, 2.5, 5.0, 2.0]]"
# [1, 0, 1, 0, 2]

curl -XGET 'localhost:5000/info/model'
# {"penalty": "l2", "tol": 0.0001, "C": 1.0, "classes_": [0, 1, 2], "coef_": [[0.4150, 1.4613, -2.2621, -1.0291], ...], ...}

curl -XGET 'localhost:5000/info/features'
# ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

curl -XGET 'localhost:5000/info/target_labels'
#  ["setosa", "versicolor", "virginica"]
```

## Advanced example: image classification with Keras

ServeIt accepts optional pre/postprocessing callback methods, making it easy start serving more complex models. Let's deploy a pre-trained Keras model to a new API endpoint so that we can classify images on the fly. We'll start by loading a ResNet50 model pre-trained on ImageNet:

```python
from keras.applications.resnet50 import ResNet50

# load Resnet50 model pretrained on ImageNet
model = ResNet50(weights='imagenet')
```

Next we define methods for loading and preprocessing an image from a URL...
```python
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from flask import request
import requests
from serveit.utils import make_serializable, get_bytes_to_image_callback

# define a loader callback for the API to fetch the relevant data and
# preprocessor callbacks to map to a format expected by the model
def loader():
    """Load image from URL, and preprocess for Resnet."""
    url = request.args.get('url')  # read image URL as a request URL param
    response = requests.get(url)  # make request to static image file
    return response.content

# get a bytes-to-image callback, resizing the image to 224x224 for ImageNet
bytes_to_image = get_bytes_to_image_callback(image_dims=(224, 224))

# create a list of different preprocessors to chain multiple steps
preprocessor = [bytes_to_image, preprocess_input]
```

... and import a decoder for postprocessing the model predictions for the API response:
```python
from keras.applications.resnet50 import decode_predictions
```

And now we're ready to start serving our image classifier:
```python
from serveit.server import ModelServer

# deploy model to a ModelServer
server = ModelServer(
    model,
    model.predict,
    data_loader=loader,
    preprocessor=preprocessor,
    postprocessor=decode_predictions,
)

# start serving
server.serve()
```

Behold:
![cat picture](img/cat.jpg)
```bash
curl -XPOST 'localhost:5000/predictions?url=https://cdn.pixabay.com/photo/2017/11/14/13/06/kitty-2948404_640.jpg'
# [[["n02123159", "tiger_cat", 0.598746120929718], ["n02127052", "lynx", 0.32807421684265137], ["n02123045", "tabby", 0.042475175112485886]]]
```

![plane picture](img/airplane.jpg)
```bash
curl -XPOST 'localhost:5000/predictions?url=https://cdn.pixabay.com/photo/2012/06/28/08/26/plane-50893_640.jpg'
# [[["n02690373", "airliner", 0.5599709749221802], ["n04592741", "wing", 0.286420077085495], ["n04552348", "warplane", 0.14331381022930145]]]
```

You can interact with a live DenseNet121 demo server at `https://imagenet-keras.ryanlee.site/predictions` (source code and sample requests [here](https://github.com/rtlee9/serveit-demo-imagenet-keras/)).

## Advanced example: serving with gunicorn
If you have a preference for a specific WSGI HTTP server, you can easily retrieve the underlying app from the server to serve separately. Once you've initialized the ModelServer class, fetch the underlying app in the global scope of a Python script like so:

```python
# main.py
app = server.get_app()
```

Now all you have to do in your shell (or Procfile) is:
```bash
# shell
gunicorn main:app

# Procfile
web: gunicorn main:app
```

[View all examples](https://github.com/rtlee9/serveit/tree/master/examples)
