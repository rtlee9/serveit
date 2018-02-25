"""Sample ServeIt Scikit-Learn server."""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from serveit.server import ModelServer

# fit a model on the Iris dataset
data = load_iris()
clf = LogisticRegression()
clf.fit(data.data, data.target)


def validator(input_data):
    """Simple model input validator.

    Validator ensures the input data array is
        - two dimensional
        - has the correct number of features.
    """
    global data
    # check num dims
    if input_data.ndim != 2:
        return False, 'Data should have two dimensions.'
    # check number of columns
    if input_data.shape[1] != data.data.shape[1]:
        reason = '{} features required, {} features provided'.format(
            data.data.shape[1], input_data.shape[1])
        return False, reason
    # validation passed
    return True, None

# deploy model to a SkLearnServer
server = ModelServer(clf, clf.predict, validator)

# add informational endpoints
server.create_info_endpoint('features', data.feature_names)
server.create_info_endpoint('target_labels', data.target_names.tolist())

# start API
server.serve()
