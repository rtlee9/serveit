"""Test PredictionServer."""
import unittest
import json

from serveit.sklearn_server import PredictionServer


class PredictionServerTest():
    """Base class to test the prediction server.

    PredictionServerTest should be inherited by a class that has a `clf` classifier
    attribute, and calls `PredictionServerTest._setup()` after instantiation.
    That class should also inherit from `unittest.TestCase` to ensure tests are executed.
    """

    def _setup(self, fit):
        """Set up method to be called before each unit test.

        Arguments:
            - fit (callable): model training method; must accept args (data, target)
        """
        from sklearn.datasets import load_iris
        self.data = load_iris()
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
        sample_data = [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3., 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5., 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
        ]
        response = self.app.post('/predictions', headers={'Content-Type': 'application/json'}, data=json.dumps(sample_data))
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        for prediction in response_data:
            self.assertIn(prediction, self.data.target)


class LogisticRegressionTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with LogisticRegression."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression()
        super(LogisticRegressionTest, self)._setup(self.clf.fit)


class SvcTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with SVC."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVC
        self.clf = SVC()
        super(SvcTest, self)._setup(self.clf.fit)


class RandomForestTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with RandomForestClassifier."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier()
        super(RandomForestTest, self)._setup(self.clf.fit)

if __name__ == '__main__':
    unittest.main()
