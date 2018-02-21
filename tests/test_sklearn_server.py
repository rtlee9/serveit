"""Test SklearnServer."""
import unittest
import json
from sklearn.datasets import load_iris

from serveit.sklearn_server import SklearnServer
from test_prediction_server import PredictionServerTest


class SklearnServerTest(PredictionServerTest):
    """Base class to test the Scikit-Learn server.

    SklearnServerTest should be inherited by a class that has a sklearn `clf`
    classifier attribute, and calls `SklearnServerTest._setup()` after instantiation.
    That class should also inherit from `unittest.TestCase` to ensure tests are executed.
    """

    def _setup(self, data):
        """Set up method to be called before each unit test."""
        super(SklearnServerTest, self)._setup(self.clf.fit, data)
        self.sklearn_server = SklearnServer(self.clf, self.clf.predict)
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


class IrisLogisticRegressionTest(unittest.TestCase, SklearnServerTest):
    """Test SklearnServer with LogisticRegression fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression()
        super(IrisLogisticRegressionTest, self)._setup(load_iris())


class IrisSvcTest(unittest.TestCase, SklearnServerTest):
    """Test SklearnServer with SVC fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVC
        self.clf = SVC()
        super(IrisSvcTest, self)._setup(load_iris())


class IrisRandomForestTest(unittest.TestCase, SklearnServerTest):
    """Test SklearnServer with RandomForestClassifier fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier()
        super(IrisRandomForestTest, self)._setup(load_iris())

if __name__ == '__main__':
    unittest.main()
