"""
ref link: https://zhuanlan.zhihu.com/p/33773140
"""

import math
import numpy as np


class Conv2D:
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        # shape = [N,W,H,C], N=batchsize, W=width, H=height, C=channels
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method
        self.col_img_i=[]
        self.col_img=[]
        # MSRA初始化
        weights_scale = math.sqrt(ksize * ksize * self.input_channels / 2)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        # padding or not
        if method == "VALID":
            self.eta = np.zeros((shape[0],math.floor ((shape[1] - ksize) / self.stride) + 1,
                                 math.floor ((shape[1] - ksize) / self.stride) + 1,
                                 self.output_channels))
        if method == 'SAME':
            self.eta = np.zeros((shape[0], math.ceil(shape[1] / self.stride),
                                 math.ceil(shape[1] / self.stride), self.output_channels))
        # gradient for bp
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, ((0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                       'constant', constant_values=0)
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_img_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_img_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_img.append(self.col_img_i)
        self.col_img = np.array(self.col_img)
        return conv_out

    def gradient(self, eta):
        # eta is error
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_img[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # conv flip kernel with padded eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1),
                (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)
        if self.method == 'SAME':
            pad_delta = np.pad(self.delta, (
                (0, 0), (self.ksize / 2, self.ksize / 2),
                (self.ksize / 2, self.ksize / 2), (0, 0)), 'constant',
                               constant_values=0)
        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array(
            [im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights = self.weights - alpha * self.w_gradient
        self.bias = self.bias - alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


def im2col(image, ksize: int, stride: int) -> np.ndarray:
    # image = [N,W,H,C], N=batchsize, W=width, H=height, C=channels
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col


# conv1 = Conv2D([2,28,28,1],12,5,1)

if __name__ == "__main__":
    # img = np.random.standard_normal((2, 32, 32, 3))
    img = np.ones((1, 32, 32, 3))
    img *= 2
    conv = Conv2D(img.shape, 12, 3, 1)
    next = conv.forward(img)
    next1 = next.copy() + 1
    conv.gradient(next1 - next)
    print(conv.w_gradient)
    print(conv.b_gradient)
    conv.backward()
