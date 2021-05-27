"""
include max pooling and avg pooling
"""
import math

import numpy as np


class Max_pooling:
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channel = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = \
            [shape[0],
             math.floor((shape[1] - ksize) / self.stride) + 1,
             math.floor((shape[1] - ksize) / self.stride) + 1,
             self.output_channel]

    def forward(self, x):
        out = np.zeros([x.shape[0],
                        math.floor((x.shape[1] - self.ksize) / self.stride) + 1,
                        math.floor((x.shape[1] - self.ksize) / self.stride) + 1,
                        self.output_channel])
        for b in range(x.shape[0]):
            for c in range(self.output_channel):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, i // self.stride, j // self.stride, c] = \
                            np.max(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i + index // self.ksize, j + index % self.ksize, c] = 1
        return out

    def backward(self, eta):
        return self.index * \
               np.repeat(np.repeat(eta,
                                   self.stride, axis=1), self.stride, axis=2)


class AVG_pooling:
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channel = shape[-1]

    def forward(self, x):
        out = np.zeros([x.shape[0],
                        math.floor((x.shape[1] - self.ksize) / self.stride) + 1,
                        math.floor((x.shape[1] - self.ksize) / self.stride) + 1,
                        self.output_channels])
        for b in range(x.shape[0]):
            for c in range(self.output_channel):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, i // self.stride, j // self.stride, c] = \
                            np.mean(x[b, i:i + self.ksize, j:j + self.ksize, c])
        return out

    def backward(self, eta):
        if self.stride == self.ksize:
            next_eta = np.repeat(eta, self.stride, axis=1)
            next_eta = np.repeat(next_eta, self.stride, axis=2)
            return next_eta / self.ksize / self.ksize
        else:
            next_eta = np.zeros(eta.shape[0], (eta.shape[1] - 1) * self.stride + self.ksize,
                                (eta.shape[2] - 1) * self.stride + self.ksize, eta.shape[3])
            for b in range(eta.shape[0]):
                for c in range(eta.shape[3]):
                    for i in range(0, eta.shape[1]):
                        for j in range(0, eta.shape[2]):
                            for k in range(0, self.ksize * self.ksize):
                                next_eta[b, i + k // self.ksize, j + k % self.ksize, c] += \
                                    (eta[b, i, j, c]) / self.ksize / self.ksize
            return next_eta
