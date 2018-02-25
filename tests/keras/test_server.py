"""Test ModelServer with Keras models."""
import unittest
from sklearn.datasets import load_boston

from tests.test_server import ModelServerTest


class BostonKerasNNTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with Keras nerual net fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        data = load_boston()
        self.model = self.get_model(data.data.shape[1])
        super(BostonKerasNNTest, self)._setup(self.model, self.model.fit, data)

    def get_model(self, input_dim):
        """Create and compile simple model."""
        from keras.models import Sequential
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(100, input_dim=input_dim, activation='sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='SGD')
        return model


if __name__ == '__main__':
    unittest.main()
