"""SkLearn model serving."""
from flask_restful import Resource
from .prediction_server import PredictionServer


class SklearnServer(PredictionServer):
    """Scikit-Learn server."""

    def __init__(self, model, predict):
        """Initialize class with model and prediction function."""
        super(SklearnServer, self).__init__(predict)
        self.model = model

    def __repr__(self):
        """String representation."""
        return '<EasyDeploySklearn: {}>'.format(self.model)

    def create_model_info_endpoint(self):
        """Create an endpoint to serve info GET requests."""
        model = self.model

        # create generic restful resource to serve model information as JSON
        class ModelInfo(Resource):
            def get(self):
                details = {}
                for key, value in model.__dict__.items():
                    try:
                        # try converting numpy values to lists
                        details[key] = value.tolist()
                    except AttributeError:
                        details[key] = value
                return details

        self.api.add_resource(ModelInfo, '/info/model')


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    # fit a model on the Iris dataset
    data = load_iris()
    reg = LogisticRegression()
    reg.fit(data.data, data.target)

    # deploy model to a SkLearnServer
    eds = SklearnServer(reg, reg.predict)

    # add informational endpoints
    eds.create_model_info_endpoint()
    eds.create_info_endpoint('features', data.feature_names)
    eds.create_info_endpoint('target_labels', data.target_names.tolist())

    # start API
    eds.serve()
