import keras
from keras.datasets import mnist
from keras.models import load_model
from scipy import signal
from keras import backend as K
import numpy as np
import tensorflow as tf


def get_mnist():
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_test, x_train, y_test, y_train


def nhwc_to_nchw(x):
    # https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
    out = np.transpose(x, [0, 3, 1, 2])
    return out

class Net:
    def __init__(self):
        self.keras_model = load_model('/home/gaspar/git/hwn/hwn/minist/mnist_nobias.h5')
        # self.keras_model.layers[0].activation = ''

    def _conv2d(self, inp, taps):
        inp = inp.astype(np.float32)
        taps = taps.astype(np.float32)
        return signal.convolve2d(inp, taps, mode='valid')

    def _relu(self, inp):
        return np.maximum(inp, 0, inp)

    def nhwc_to_nchw(self, x):
        out = tf.transpose(x, [0, 3, 1, 2])
        return out

    def keras_layer_output(self, inp, layer_i):
        layer = self.keras_model.layers[layer_i]

        # calculate keras output
        output = K.function([layer.input, K.learning_phase()], [layer.output])

        learning_phase = 0
        layer_output = output([inp, learning_phase])[0]
        layer_output = nhwc_to_nchw(layer_output)
        return layer_output

    def layer_output(self, inp, layer_i):
        inp = np.squeeze(inp)
        layer = self.keras_model.layers[layer_i]
        taps = np.squeeze(layer.get_weights()[0])
        taps = np.swapaxes(taps, 0, 2)
        # taps = np.swapaxes(taps, 1, 2)
        print(taps.shape)

        out = []
        for i, filter in enumerate(taps):
            if i == 0:
                print('orig ', filter)
                print('T ', filter.T)
                print(np.rot90(filter, 2).T)
            # filter = filter.T
            # print(filter)
            # filter = np.flip(filter, 1).T
            filter = np.rot90(filter, 2).T
            # filter = np.swapaxes(filter, 0, 1) * 0.1
            outp = self._conv2d(inp, filter)
            out.append(outp)

        out = np.array(out)
        # out = self._relu(out)
        return out

