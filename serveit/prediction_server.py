"""Base class for serving predictions."""
from flask import Flask, request
from flask_restful import Resource, Api

from .config import WSGI_OPTIONS
from .wsgi import StandaloneApplication
from .utils import make_serializable


class PredictionServer(object):
    """Easy deploy class."""

    def __init__(self, predict):
        """Initialize class with prediction function."""
        self.predict = predict
        self.app = Flask('{}_{}'.format(self.__class__.__name__, type(predict).__name__))
        self.api = Api(self.app)
        self._create_prediction_endpoint()

    def __repr__(self):
        """String representation."""
        return '<PredictionsServer: {}>'.format(type(self.predict).__name__)

    def _create_prediction_endpoint(self):
        """Create an endpoint to serve predictions."""
        # copy instance variables to local scope for resource class
        predict = self.predict
        logger = self.app.logger

        # create restful resource
        class Predictions(Resource):
            def post(self):
                data = request.get_json()
                logger.debug(data)
                prediction = predict(data)
                logger.debug(prediction)
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

    def serve(self):
        """Serve predictions as an API endpoint."""
        self.server = StandaloneApplication(self.app, WSGI_OPTIONS).run()
