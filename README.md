# ServeIt
[![Build Status](https://travis-ci.org/rtlee9/serveit.svg?branch=master)](https://travis-ci.org/rtlee9/serveit)
[![Codacy Grade Badge](https://api.codacy.com/project/badge/Grade/2af32a3840d5441e815f3956659b091f)](https://www.codacy.com/app/ryantlee9/serveit)
[![Codacy Coverage Badge](https://api.codacy.com/project/badge/Coverage/2af32a3840d5441e815f3956659b091f)](https://www.codacy.com/app/ryantlee9/serveit)
[![PyPI version](https://badge.fury.io/py/ServeIt.svg)](https://badge.fury.io/py/ServeIt)

ServeIt lets you serve model predictions and supplementary information from a RESTful API in one line of code. Current features include:

1. Model prediction serving
1. Supplementary information endpoint creation
1. User-provided input validation and exception handling
1. Configurable request and response logging (work in progress)


## Installation: Python 2.7 and Python 3.6
Installation is easy with pip: `pip install serveit`

## Usage:
Deploy your model to an API endpoint with one line of code:
```python
from serveit.server import ModelServer

# initialize the server with a model and a method to use
# for predictions
ModelServer(clf, clf.predict).serve()
```

Then check out your new API:
```bash
curl -XPOST 'localhost:5000/predictions'\
	-H "Content-Type: application/json"\
	-d "[[5.6, 2.9, 3.6, 1.3], [4.4, 2.9, 1.4, 0.2], [5.5, 2.4, 3.8, 1.1], [5.0, 3.4, 1.5, 0.2], [5.7, 2.5, 5.0, 2.0]]"
# [1, 0, 1, 0, 2]
```

Please see the [examples](examples) directory for additional usage.

## Supported libraries
* Scikit-Learn
* Keras

## Coming soon:
* TensorFlow
* PyTorch
