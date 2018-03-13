# ServeIt
[![Build Status](https://travis-ci.org/rtlee9/serveit.svg?branch=master)](https://travis-ci.org/rtlee9/serveit)
[![Codacy Grade Badge](https://api.codacy.com/project/badge/Grade/2af32a3840d5441e815f3956659b091f)](https://www.codacy.com/app/ryantlee9/serveit)
[![Codacy Coverage Badge](https://api.codacy.com/project/badge/Coverage/2af32a3840d5441e815f3956659b091f)](https://www.codacy.com/app/ryantlee9/serveit)
[![PyPI version](https://badge.fury.io/py/ServeIt.svg)](https://badge.fury.io/py/ServeIt)

ServeIt lets you serve model predictions and supplementary information from a RESTful API on any domain using your favorite ML library in as little as one line of code:

```python
from serveit.server import ModelServer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# fit logistic regression on Iris data
clf = LogisticRegression()
data = load_iris()
clf.fit(data.data, data.target)

# initialize server with a model and start serving predictions
ModelServer(clf, clf.predict).serve()
```

Your new API is now accepting `POST` requests at `localhost:5000/predictions`! Please see the [examples](examples) directory for detailed examples across domains (e.g., regression, image classification), including live examples.

#### Features
Current ServeIt features include:

1. Model inference serving via RESTful API endpoint
1. Extensible library for inference-time data loading, preprocessing, input validation, and postprocessing
1. Supplementary information endpoint creation
1. Automatic JSON serialization of responses
1. Configurable request and response logging (work in progress)

#### Supported libraries
The following libraries are currently supported:
* Scikit-Learn
* Keras
* PyTorch

## Installation: Python 2.7 and Python 3.6
Installation is easy with pip: `pip install serveit`

## Building
You can build locally with: `python setup.py`

## License
[MIT](LICENSE.md)
