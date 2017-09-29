import keras
import numpy as np


class WeightShaper(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.cooldown = 0

    def callback(self):
        def scale_filter_output(layer):
            c = layer.get_weights()[0]
            cs = c.reshape(list(reversed(c.shape)))

            for i, xx in enumerate(cs):
                for j, x in enumerate(xx):
                    cs[i][j] = x / np.sum(np.abs(x))

            css = cs.reshape(c.shape)
            layer.set_weights([css])

        scale_filter_output(self.model.layers[0])
        scale_filter_output(self.model.layers[2])
        scale_filter_output(self.model.layers[4])
        scale_filter_output(self.model.layers[6])
        scale_filter_output(self.model.layers[8])
        scale_filter_output(self.model.layers[10])

    def on_train_begin(self, logs={}):
        self.callback()

    def on_batch_end(self, batch, logs={}):
        self.cooldown += 1
        if self.cooldown > 64:
            self.callback()
            self.cooldown = 0
            print('JO')
