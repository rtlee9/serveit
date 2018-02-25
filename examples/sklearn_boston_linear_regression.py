"""Sample ServeIt Scikit-Learn server."""
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from serveit.server import ModelServer

# fit a model on the Boston housing dataset
data = load_boston()
reg = LinearRegression()
reg.fit(data.data, data.target)


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
server = ModelServer(reg, reg.predict, validator)

# add informational endpoints
server.create_info_endpoint('features', data.feature_names)

# start API
server.serve()
