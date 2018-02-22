"""Base class for serving predictions."""
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import numpy as np
from meinheld import server, middleware

from .utils import make_serializable
from .log_utils import get_logger

logger = get_logger(__name__)


class PredictionServer(object):
    """Easy deploy class."""

    def __init__(self, predict, input_validation=lambda data: (True, None)):
        """Initialize class with prediction function.

        Arguments:
            - predict (fn): function that takes a numpy array of features as input,
                and returns a prediction of targets
            - input_validation (fn): takes a numpy array as input;
                returns True if validation passes and False otherwise
        """
        self.predict = predict
        self.app = Flask('{}_{}'.format(self.__class__.__name__, type(predict).__name__))
        self.api = Api(self.app, catch_all_404s=True)
        self._create_prediction_endpoint(input_validation)
        self.app.logger.setLevel(logger.level)  # TODO: separate configuration for API loglevel

    def __repr__(self):
        """String representation."""
        return '<PredictionsServer: {}>'.format(type(self.predict).__name__)

    def _create_prediction_endpoint(self, input_validation=lambda data: (True, None)):
        """Create an endpoint to serve predictions.

        Arguments:
            - input_validation (fn): takes a numpy array as input;
                returns True if validation passes and False otherwise
        """
        # copy instance variables to local scope for resource class
        predict = self.predict
        logger = self.app.logger

        # create restful resource
        class Predictions(Resource):
            def post(self):
                # parse request data
                data = request.get_json()
                logger.debug('Received JSON data of length {:,}'.format(len(data)))

                # convert to numpy array
                data = np.array(data)
                logger.debug('Converted JSON data to Numpy array with shape {}'.format(data.shape))

                # sanity check using user defined callback (default is no check)
                validation_pass, validation_reason = input_validation(data)
                if not validation_pass:
                    # if validation fails, log the reason code, log the data, and send a 400 response
                    validation_message = 'Input validation failed with reason: {}'.format(validation_reason)
                    logger.error(validation_message)
                    logger.debug('Data: {}'.format(data))
                    response = jsonify(dict(message=validation_message))
                    response.status_code = 400
                    return response

                try:
                    prediction = predict(data)
                except Exception as e:
                    # log exception and return the message in a 500 response
                    logger.error('{} exception: {}'.format(type(e).__name__, e))
                    response = jsonify(dict(message=str(e)))
                    response.status_code = 500
                    return response
                logger.debug('Predictions generated with shape {}'.format(prediction.shape))
                return prediction.tolist()

        # map resource to endpoint
        self.api.add_resource(Predictions, '/predictions')

    def create_info_endpoint(self, name, data):
        """Create an endpoint to serve info GET requests."""
        # make sure data is serializable
        data = make_serializable(data)

        # create generic restful resource to serve static JSON data
        class InfoBase(Resource):
            def get(self):
                return data

        def info_factory(name):
            """Return an Info derivative resource."""
            class NewClass(InfoBase):
                pass
            NewClass.__name__ = "{}_{}".format(name, InfoBase.__name__)
            return NewClass

        self.api.add_resource(info_factory(name), '/info/{}'.format(name))

    def serve(self, host='127.0.0.1', port=5000):
        """Serve predictions as an API endpoint."""
        # self.app.run(host=host, port=port)
        server.listen((host, port))
        server.run(middleware.WebSocketMiddleware(self.app))
