# ServeIt
[![Build Status](https://travis-ci.org/rtlee9/serveit.svg?branch=master)](https://travis-ci.org/rtlee9/serveit)
[![Codacy Grade Badge](https://api.codacy.com/project/badge/Grade/2af32a3840d5441e815f3956659b091f)](https://www.codacy.com/app/ryantlee9/serveit)
[![Codacy Coverage Badge](https://api.codacy.com/project/badge/Coverage/2af32a3840d5441e815f3956659b091f)](https://www.codacy.com/app/ryantlee9/serveit)
[![PyPI version](https://badge.fury.io/py/ServeIt.svg)](https://badge.fury.io/py/ServeIt)

ServeIt lets you easily serve model predictions and supplementary information from a RESTful API. Current features include:

1. Model inference serving via RESTful API endpoint
1. Extensible library for inference-time data loading, preprocessing, input validation, and postprocessing
1. Supplementary information endpoint creation
1. Automatic JSON serialization of responses
1. Configurable request and response logging (work in progress)

## Installation: Python 2.7 and Python 3.6
Installation is easy with pip: `pip install serveit`

## Usage:
Deploy your model `clf` to an API endpoint with as little as one line of code:
```python
from serveit.server import ModelServer

# initialize server with a model and a method to use for predictions
# then start serving predictions
ModelServer(clf, clf.predict).serve()
```

Your new API is now accepting `POST` requests at `localhost:5000/predictions`! Please see the [examples](examples) directory for additional usage.

### Supported libraries
* Scikit-Learn
* Keras
* PyTorch

### Coming soon:
* TensorFlow

## Building
You can build locally with: `python setup.py`

## License
[MIT](LICENSE.md)
