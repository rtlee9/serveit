"""Test SklearnServer."""
import unittest
import json
from sklearn.datasets import load_iris, load_boston

from serveit.sklearn_server import SklearnServer
from tests.test_prediction_server import PredictionServerTest


class SklearnServerTest(PredictionServerTest, object):
    """Base class to test the Scikit-Learn server.

    SklearnServerTest should be inherited by a class that has a sklearn `model`
    attribute, and calls `SklearnServerTest._setup()` after instantiation.
    That class should also inherit from `unittest.TestCase` to ensure tests are executed.
    """

    def _setup(self, data):
        """Set up method to be called before each unit test."""
        super(SklearnServerTest, self)._setup(self.model.fit, data)
        self.sklearn_server = SklearnServer(self.model, self.model.predict)
        self.app = self.sklearn_server.app.test_client()

    def test_model_info_none(self):
        """Verify 404 response if /info/model endpoint not yet created."""
        response = self.app.get('/info/model')
        self.assertEqual(response.status_code, 404)

    def test_model_info(self):
        """Test model info endpoint."""
        self.sklearn_server.create_model_info_endpoint()
        app = self.sklearn_server.app.test_client()
        response = app.get('/info/model')
        response_data = json.loads(response.get_data())
        self.assertGreater(len(response_data), 3)
        expected_keys = ['classes_']
        for key in expected_keys:
            self.assertIn(key, response_data)

    def test_input_validation(self):
        """Add simple input validator and make sure it triggers."""
        import numpy as np

        # model input validator
        def feature_count_check(data):
            # check num dims
            if data.ndim != 2:
                return False, 'Data should have two dimensions.'
            # check number of columns
            if data.shape[1] != self.data.data.shape[1]:
                reason = '{} features required, {} features provided'.format(
                    data.shape[1], self.data.data.shape[1])
                return False, reason
            # validation passed
            return True, None

        # set up test server
        sklearn_server = SklearnServer(self.model, self.model.predict, feature_count_check)
        app = sklearn_server.app.test_client()

        # generate sample data
        sample_idx = np.random.randint(self.data.data.shape[0], size=100)
        sample_data = self.data.data[sample_idx, :]

        # post good data, verify 200 response
        response = app.post(
            '/predictions',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_data.tolist()),
        )
        self.assertEqual(response.status_code, 200)

        # post bad data (drop a single column), verify 400 response
        response = app.post(
            '/predictions',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_data[:, :-1].tolist()),
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.get_data())
        expected_reason = '{} features required, {} features provided'.format(
            self.data.data.shape[1] - 1, self.data.data.shape[1])
        self.assertIn(expected_reason, response_data['message'])


class IrisLogisticRegressionTest(unittest.TestCase, SklearnServerTest):
    """Test SklearnServer with LogisticRegression fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        super(IrisLogisticRegressionTest, self)._setup(load_iris())


class IrisSvcTest(unittest.TestCase, SklearnServerTest):
    """Test SklearnServer with SVC fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVC
        self.model = SVC()
        super(IrisSvcTest, self)._setup(load_iris())


class IrisRandomForestTest(unittest.TestCase, SklearnServerTest):
    """Test SklearnServer with RandomForestClassifier fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        super(IrisRandomForestTest, self)._setup(load_iris())


class BostonLinearRegressionTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with LogisticRegression fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        super(BostonLinearRegressionTest, self)._setup(self.model.fit, load_boston())


class BostonSvrTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with SVR fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVR
        self.model = SVR()
        super(BostonSvrTest, self)._setup(self.model.fit, load_boston())


class BostonRfrTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with LogisticRegression fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor()
        super(BostonRfrTest, self)._setup(self.model.fit, load_boston())

if __name__ == '__main__':
    unittest.main()
