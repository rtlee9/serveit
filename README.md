# ServeIt
[![Build Status](https://travis-ci.org/rtlee9/serveit.svg?branch=master)](https://travis-ci.org/rtlee9/serveit)
[![PyPI version](https://badge.fury.io/py/ServeIt.svg)](https://badge.fury.io/py/ServeIt)
[![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)](#installation-python-27-and-python-36)
[![Python 3.7](https://img.shields.io/badge/python-3.6-blue.svg)](#installation-python-27-and-python-36)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


ServeIt deploys your trained models to a RESTful API for prediction serving. Current features include:

1. Model prediction serving
1. Supplementary information endpoint creation
1. Configurable request and response logging (work in progress)


## Installation: Python 2.7 and Python 3.6
* PyPi: `pip install serveit`
* source: `git clone https://github.com/rtlee9/serveit.git && cd serveit && pip install -e .`  # WIP

## Supported libraries
* Scikit-Learn

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from serveit.sklearn_server import SklearnServer

# fit a model on the Iris dataset
data = load_iris()
reg = LogisticRegression()
reg.fit(data.data, data.target)

# deploy model to a SkLearnServer
sklearn_server = SklearnServer(reg, reg.predict)

# add informational endpoints
sklearn_server.create_model_info_endpoint()
sklearn_server.create_info_endpoint('features', data.feature_names)
sklearn_server.create_info_endpoint('target_labels', data.target_names.tolist())

# start API
sklearn_server.serve()
```

Then try out your new API:
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

## Coming soon:
* TensorFlow
* Keras
* PyTorch
