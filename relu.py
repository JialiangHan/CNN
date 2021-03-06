import numpy as np


class Relu:
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, eta):
        self.eta = eta
        self.eta[self.x < 0] = 0
        return self.eta
