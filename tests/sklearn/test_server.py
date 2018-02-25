"""Test ModelServer with Scikit-Learn models."""
import unittest
from sklearn.datasets import load_iris, load_boston

from tests.test_server import ModelServerTest


class IrisLogisticRegressionTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with LogisticRegression fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        super(IrisLogisticRegressionTest, self)._setup(self.model, self.model.fit, load_iris())


class IrisSvcTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with SVC fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVC
        self.model = SVC()
        super(IrisSvcTest, self)._setup(self.model, self.model.fit, load_iris())


class IrisRfcTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with RandomForestClassifier fitted on iris data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()
        super(IrisRfcTest, self)._setup(self.model, self.model.fit, load_iris())


class BostonLinearRegressionTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with LogisticRegression fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        super(BostonLinearRegressionTest, self)._setup(self.model, self.model.fit, load_boston())


class BostonSvrTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with SVR fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.svm import SVR
        self.model = SVR()
        super(BostonSvrTest, self)._setup(self.model, self.model.fit, load_boston())


class BostonRfrTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with LogisticRegression fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor()
        super(BostonRfrTest, self)._setup(self.model, self.model.fit, load_boston())

if __name__ == '__main__':
    unittest.main()
