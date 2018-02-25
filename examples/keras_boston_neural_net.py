"""Sample ServeIt prediction server."""
from sklearn.datasets import load_boston
from serveit.server import ModelServer
from keras.models import Sequential
from keras.layers import Dense


def get_model(input_dim):
    """Create and compile simple model."""
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='SGD')
    return model

# fit a model on the Boston housing dataset
data = load_boston()
model = get_model(data.data.shape[1])
model.fit(data.data, data.target)


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

# deploy model to a ModelServer
server = ModelServer(model, model.predict, validator)

# add informational endpoints
server.create_info_endpoint('features', data.feature_names)

# start API
server.serve()
