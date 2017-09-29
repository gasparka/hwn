import keras
import numpy as np

import logging

from keras.layers import Conv2D

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

            coefs = []
            total_error = 0
            for i, xx in enumerate(cs):
                for j, x in enumerate(xx):
                    coef = np.sum(np.abs(x)) / self.max_output
                    coefs.append(coef)
                    if coef > 1.1:
                        coef = 1.1
                    if coef < 0.9:
                        coef = 0.9
                    cs[i][j] = x / coef
                    total_error += coef - 1

            log.info('Coef AVG={} MAX={} MIN={}'.format(np.mean(coefs), np.array(coefs).max(), np.array(coefs).min()))
            css = cs.reshape(c.shape)
            layer.set_weights([css])
            return total_error

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, Conv2D) and layer.get_weights()[0].shape[:2] == (3, 3):
                err = scale_filter_output(layer)
                log.info('Adjusted layer {} weights, error was {}\n'.format(i, err))


    def on_train_begin(self, logs={}):
        self.callback()

    def on_batch_end(self, batch, logs={}):
        self.cooldown_counter += 1
        if self.cooldown_counter % self.cooldown == 0:
            self.callback()
