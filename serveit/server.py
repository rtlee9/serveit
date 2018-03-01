"""Base class for serving predictions."""
from flask import Flask, jsonify
from flask_restful import Resource, Api
import numpy as np

from .utils import make_serializable, json_numpy_loader
from .log_utils import get_logger

logger = get_logger(__name__)


def exception_log_and_respond(exception, logger, message, status_code):
    """Log an error and send jsonified respond."""
    logger.error('{} exception: {}'.format(type(exception).__name__, exception))
    return make_response(
        message,
        status_code,
        dict(exception_type=type(exception).__name__, exception_message=str(exception)),
    )


def make_response(message, status_code, details=None):
    """Make a jsonified response with specified message and status code."""
    response_body = dict(message=message)
    if details:
        response_body['details'] = details
    response = jsonify(response_body)
    response.status_code = status_code
    return response


class ModelServer(object):
    """Easy deploy class."""

    def __init__(
            self,
            model,
            predict,
            input_validation=lambda data: (True, None),
            data_loader=json_numpy_loader,
            preprocessor=lambda x: x,
            postprocessor=make_serializable):
        """Initialize class with prediction function.

        Arguments:
            - predict (fn): function that takes a numpy array of features as input,
                and returns a prediction of targets
            - input_validation (fn): takes a numpy array as input;
                returns True if validation passes and False otherwise
            - data_loader (fn): reads flask request and returns data preprocessed to be
                used in the `predict` method
            - postprocessor (fn): transforms the predictions from the `predict` method
        """
        self.model = model
        self.predict = predict
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.app = Flask('{}_{}'.format(self.__class__.__name__, type(predict).__name__))
        self.api = Api(self.app, catch_all_404s=True)
        self._create_prediction_endpoint(
            data_loader=data_loader,
            input_validation=input_validation,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        logger.info('Model predictions registered to endpoint /predictions (available via POST)')
        self.app.logger.setLevel(logger.level)  # TODO: separate configuration for API loglevel
        self._create_model_info_endpoint()

    def __repr__(self):
        """String representation."""
        return '<PredictionsServer: {}>'.format(type(self.predict).__name__)

    def _create_prediction_endpoint(
            self,
            to_numpy=True,
            data_loader=json_numpy_loader,
            preprocessor=lambda x: x,
            input_validation=lambda data: (True, None),
            postprocessor=make_serializable):
        """Create an endpoint to serve predictions.

        Arguments:
            - input_validation (fn): takes a numpy array as input;
                returns True if validation passes and False otherwise
            - data_loader (fn): reads flask request and returns data preprocessed to be
                used in the `predict` method
            - postprocessor (fn): transforms the predictions from the `predict` method
        """
        # copy instance variables to local scope for resource class
        predict = self.predict
        logger = self.app.logger

        # create restful resource
        class Predictions(Resource):
            @staticmethod
            def post():
                # read data from API request
                try:
                    data = data_loader()
                except Exception as e:
                    return exception_log_and_respond(e, logger, 'Unable to fetch data', 400)

                data = preprocessor(data)  # preprocess data
                data = np.array(data) if to_numpy else data  # convert to numpy

                # sanity check using user defined callback (default is no check)
                validation_pass, validation_reason = input_validation(data)
                if not validation_pass:
                    # if validation fails, log the reason code, log the data, and send a 400 response
                    validation_message = 'Input validation failed with reason: {}'.format(validation_reason)
                    logger.error(validation_message)
                    logger.debug('Data: {}'.format(data))
                    return make_response(validation_message, 400)

                try:
                    prediction = predict(data)
                except Exception as e:
                    # log exception and return the message in a 500 response
                    logger.debug('Data: {}'.format(data))
                    return exception_log_and_respond(e, logger, 'Unable to make prediction', 500)
                logger.debug(prediction)
                logger.debug('Predictions generated with shape {}'.format(prediction.shape))
                return postprocessor(prediction)

        # map resource to endpoint
        self.api.add_resource(Predictions, '/predictions')

    def create_info_endpoint(self, name, data):
        """Create an endpoint to serve info GET requests."""
        # make sure data is serializable
        data = make_serializable(data)

        # create generic restful resource to serve static JSON data
        class InfoBase(Resource):
            @staticmethod
            def get():
                return data

        def info_factory(name):
            """Return an Info derivative resource."""
            class NewClass(InfoBase):
                pass
            NewClass.__name__ = "{}_{}".format(name, InfoBase.__name__)
            return NewClass

        path = '/info/{}'.format(name)
        self.api.add_resource(info_factory(name), path)
        logger.info('Regestered informational resource to {} (available via GET)'.format(path))
        logger.debug('Endpoint {} will now serve the following static data:\n{}'.format(path, data))

    def _create_model_info_endpoint(self, path='/info/model'):
        """Create an endpoint to serve info GET requests."""
        model = self.model

        # parse model details
        model_details = {}
        for key, value in model.__dict__.items():
            model_details[key] = make_serializable(value)

        # create generic restful resource to serve model information as JSON
        class ModelInfo(Resource):
            @staticmethod
            def get():
                return model_details

        self.api.add_resource(ModelInfo, path)
        self.app.logger.info('Regestered informational resource to {} (available via GET)'.format(path))
        self.app.logger.debug('Endpoint {} will now serve the following static data:\n{}'.format(path, model_details))

    def serve(self, host='127.0.0.1', port=5000):
        """Serve predictions as an API endpoint."""
        from meinheld import server, middleware
        # self.app.run(host=host, port=port)
        server.listen((host, port))
        server.run(middleware.WebSocketMiddleware(self.app))

    def get_app(self):
        """Return the underlying Flask app."""
        return self.app
