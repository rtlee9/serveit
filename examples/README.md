# ServeIt examples

Start by fitting a model:
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# fit a model on the Iris dataset
data = load_iris()
clf = LogisticRegression()
clf.fit(data.data, data.target)
```
Serve your trained model:
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

Then check out your new API:
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

[View all examples](https://github.com/rtlee9/serveit/tree/master/examples)
