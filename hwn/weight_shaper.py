import keras
import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('log')


class WeightShaper(keras.callbacks.Callback):
    def __init__(self, max_output, cooldown=16):
        super().__init__()
        log.info('Using weight shaper max_output: {}'.format(max_output))
        self.max_output = max_output
        self.cooldown = cooldown
        self.cooldown_counter = 0

    def callback(self):
        def scale_filter_output(layer):
            c = layer.get_weights()[0]
            cs = c.reshape(list(reversed(c.shape)))

            total_error = 0
            for i, xx in enumerate(cs):
                for j, x in enumerate(xx):
                    coef = np.sum(np.abs(x)) / self.max_output
                    cs[i][j] = x / coef
                    total_error += coef - 1

            css = cs.reshape(c.shape)
            layer.set_weights([css])
            return total_error

        total_error = 0
        total_error += scale_filter_output(self.model.layers[0])
        total_error += scale_filter_output(self.model.layers[2])
        total_error += scale_filter_output(self.model.layers[4])
        total_error += scale_filter_output(self.model.layers[6])
        total_error += scale_filter_output(self.model.layers[8])
        total_error += scale_filter_output(self.model.layers[10])
        log.info('Adjusted weights, total error was: {}'.format(total_error))

    def on_train_begin(self, logs={}):
        self.callback()

    def on_batch_end(self, batch, logs={}):
        self.cooldown_counter += 1
        if self.cooldown_counter % self.cooldown == 0:
            self.callback()
