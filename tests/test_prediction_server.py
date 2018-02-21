"""Test PredictionServer."""
import unittest
import json
import numpy as np
from sklearn.datasets import load_iris, load_boston

from serveit.sklearn_server import PredictionServer


class PredictionServerTest(object):
    """Base class to test the prediction server.

    PredictionServerTest should be inherited by a class that has a `clf` classifier
    attribute, and calls `PredictionServerTest._setup()` after instantiation.
    That class should also inherit from `unittest.TestCase` to ensure tests are executed.
    """

    def _setup(self, fit, data):
        """Set up method to be called before each unit test.

        Arguments:
            - fit (callable): model training method; must accept args (data, target)
        """
        self.data = data
        fit(self.data.data, self.data.target)
        self.sklearn_server = PredictionServer(self.clf.predict)
        self.app = self.sklearn_server.app.test_client()

    def test_features_info_none(self):
        """Verify 404 response if '/info/features' endpoint not yet created."""
        response = self.app.get('/info/features')
        self.assertEqual(response.status_code, 404)

    def test_features_info(self):
        """Test features info endpoint."""
        self.sklearn_server.create_info_endpoint('features', self.data.feature_names)
        app = self.sklearn_server.app.test_client()
        response = app.get('/info/features')
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), self.data.data.shape[1])
        try:
            self.assertCountEqual(response_data, self.data.feature_names)
        except AttributeError:  # Python 2
            self.assertItemsEqual(response_data, self.data.feature_names)

    def test_target_labels_info_none(self):
        """Verify 404 response if '/info/target_labels' endpoint not yet created."""
        response = self.app.get('/info/target_labels')
        self.assertEqual(response.status_code, 404)

    def test_target_labels_info(self):
        """Test target labels info endpoint."""
        self.sklearn_server.create_info_endpoint('target_labels', self.data.target_names.tolist())
        app = self.sklearn_server.app.test_client()
        response = app.get('/info/target_labels')
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), self.data.target_names.shape[0])
        try:
            self.assertCountEqual(response_data, self.data.target_names)
        except AttributeError:  # Python 2
            self.assertItemsEqual(response_data, self.data.target_names)

    def test_predictions(self):
        """Test predictions endpoint."""
        sample_idx = np.random.randint(self.data.data.shape[0], size=10)
        sample_data = self.data.data[sample_idx, :]
        response = self.app.post(
            '/predictions',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_data.tolist()),
        )
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        for prediction in response_data:
            self.assertIn(prediction, self.data.target)


class IrisLogisticRegressionTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with LogisticRegression fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression()
        super(IrisLogisticRegressionTest, self)._setup(self.clf.fit, load_iris())


class IrisSvcTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with SVC fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVC
        self.clf = SVC()
        super(IrisSvcTest, self)._setup(self.clf.fit, load_iris())


class IrisRandomForestTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with RandomForestClassifier fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier()
        super(IrisRandomForestTest, self)._setup(self.clf.fit, load_iris())


if __name__ == '__main__':
    unittest.main()
