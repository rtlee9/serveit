"""Test PredictionServer."""
import unittest
import json
import numpy as np
from sklearn.datasets import load_iris, load_boston

from serveit.sklearn_server import PredictionServer


class PredictionServerTest(object):
    """Base class to test the prediction server.

    PredictionServerTest should be inherited by a class that has a `model` attribute,
    and calls `PredictionServerTest._setup()` after instantiation. That class should
    also inherit from `unittest.TestCase` to ensure tests are executed.
    """

    def _setup(self, fit, data):
        """Set up method to be called before each unit test.

        Arguments:
            - fit (callable): model training method; must accept args (data, target)
        """
        self.data = data
        fit(self.data.data, self.data.target)
        self.sklearn_server = PredictionServer(self.model.predict)
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
        if not hasattr(self.data, 'target_names'):
            return
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
        sample_idx = np.random.randint(self.data.data.shape[0], size=100)
        sample_data = self.data.data[sample_idx, :]
        response = self.app.post(
            '/predictions',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(sample_data.tolist()),
        )
        response_data = json.loads(response.get_data())
        self.assertEqual(len(response_data), len(sample_data))
        if self.data.target.ndim > 1:
            # for multiclass each prediction should be one of the training labels
            for prediction in response_data:
                self.assertIn(prediction, self.data.target)
        else:
            # the average regression prediction for a sample of data should be similar
            # to the population mean
            # TODO: remove variance from this test (i.e., no chance of false negative)
            pred_pct_diff = np.array(response_data).mean() / self.data.target.mean() - 1
            self.assertAlmostEqual(pred_pct_diff / 1e4, 0, places=1)


class IrisLogisticRegressionTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with LogisticRegression fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        super(IrisLogisticRegressionTest, self)._setup(self.model.fit, load_iris())


class IrisSvcTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with SVC fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVC
        self.model = SVC()
        super(IrisSvcTest, self)._setup(self.model.fit, load_iris())


class IrisRfcTest(unittest.TestCase, PredictionServerTest):
    """Test PredictionServer with RandomForestClassifier fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        super(IrisRfcTest, self)._setup(self.model.fit, load_iris())


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
