"""Test ModelServer with Keras models."""
import unittest
from sklearn.datasets import load_boston
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np

from tests.test_server import ModelServerTest


class BostonPytorchNNTest(unittest.TestCase, ModelServerTest):
    """Test ModelServer with Keras nerual net fitted on housing data."""

    def setUp(self):
        """Unittest set up."""
        data = load_boston()
        self.model = self.get_model(data.data.shape[1], 1)

        #  define preprocessing callback chain
        preprocessor = [
            np.array,  # convert to numpy array
            torch.from_numpy,  # convert to tensor
            lambda tensor: tensor.type(torch.FloatTensor),  # convert to FloatTensor
            torch.autograd.Variable,  # convert to PyTorch Variable
        ]

        super(BostonPytorchNNTest, self)._setup(
            self.model, self.train, data, preprocessor=preprocessor, predict=self.model, to_numpy=False, postprocessor=lambda variable: variable.data.numpy())

    def train(self, x, y):
        loss = torch.nn.MSELoss(size_average=True)
        optimizer = optim.SGD(self.model.parameters(), lr=1e-8)
        x = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=False)
        y = Variable(torch.from_numpy(y).type(torch.FloatTensor), requires_grad=False)

        for i in range(100):
            # Reset gradient
            optimizer.zero_grad()

            # Forward
            fx = self.model.forward(x)
            output = loss.forward(fx, y)

            # Backward
            output.backward()

            # Update parameters
            optimizer.step()

        return output.data[0]

    @staticmethod
    def get_model(input_dim, output_dim):
        """Create and compile simple model."""
        model = torch.nn.Sequential()
        model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=False))
        return model


if __name__ == '__main__':
    unittest.main()
