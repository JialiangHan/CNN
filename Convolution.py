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
        # MSRA初始化
        weights_scale = math.sqrt(ksize * ksize * self.input_channels / 2)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        # padding or not
        if method == "VALID":
            self.eta = np.zeros((shape[0], (shape[1] - ksize) / self.stride + 1, (shape[2] - ksize) / self.stride + 1,
                                 self.output_channels))
        if method == 'SAME':
            self.eta - np.zeros((shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels))
        # gradient for bp
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

    def forward(self,x):


def im2clo(image, ksize,stride):

# conv1 = Conv2D([2,28,28,1],12,5,1)
