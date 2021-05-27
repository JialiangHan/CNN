import numpy as np

from Convolution import Conv2D
from FC import Fully_Connect
from Pooling import Max_pooling, AVG_pooling
from Softmax import Softmax
from load_mnist import load_mnist
from relu import Relu

images, labels = load_mnist('./mnist')
test_images, test_labels = load_mnist('./mnist', 't10k')

batch_size = 64
conv1 = Conv2D([batch_size, 28, 28, 1], 12, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = Max_pooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = Max_pooling(relu2.output_shape)
fc = Fully_Connect(pool2.output_shape, 10)
sf = Softmax(fc.output_shape)

for epoch in range(20):
    learning_rate = 1e-4
    # training
    for i in range(images.shape[0] / batch_size):
        # forward
        img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = conv1.forward(img)
        relu1_out = relu1.forward(conv1_out)
        pool1_out = pool1.forward(relu1_out)
        conv2_out = conv2.forward(pool1_out)
        relu2_out = relu2.forward(conv2_out)
        pool2_out = pool2.forward(relu2_out)
        fc_out = fc.forward(pool2_out)

        print("loss: %10.3f" % sf.cal_loss(fc_out, np.array(label)))

        # backward
        sf.backward()
        fc_back = fc.backward(sf.eta)
        pool2_back = pool2.backward(fc_back)
        relu2_back = relu2.backward(pool2_back)
        conv2_back = conv2.backward(relu2_back)
        pool1_back = pool1.backward(conv2_back)
        relu1_back = relu1.backward(pool1_back)
        conv1_back = conv1.backward(relu1_back)