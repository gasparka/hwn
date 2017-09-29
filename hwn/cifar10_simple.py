# adapted from https://github.com/MateLabs/All-Conv-Keras/blob/master/allconv.py
from __future__ import print_function

import pickle
from datetime import datetime

import fire
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Conv2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from hwn.weight_shaper import WeightShaper


class Cifar10Simple:
    def __init__(self, use_bias=True, activation='relu', epochs=60, use_weight_shaper=False, weight_shaper_max_output=1.0, weight_shaper_cooldown=16):
        self.weight_shaper_cooldown = weight_shaper_cooldown
        self.weight_shaper_max_output = weight_shaper_max_output
        self.use_weight_shaper = use_weight_shaper
        self.epochs = epochs
        self.activation = activation
        self.use_bias = use_bias
        self.id_str = datetime.now().strftime("%Y%m%d%H%M%S%f") + \
                      '_use_bias={}_activation={}_epochs={}_use_weight_shaper={}'.format(self.use_bias,
                                                                                        self.activation,
                                                                                        self.epochs,
                                                                                        self.use_weight_shaper)

        if use_weight_shaper and not use_bias:
            Exception('Will not work...set use_bias=False')

    def keras_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), use_bias=self.use_bias, padding='same', input_shape=(32, 32, 3)))
        model.add(Activation(self.activation))
        model.add(Conv2D(32, (3, 3), use_bias=self.use_bias, padding='same', strides=(2, 2)))
        model.add(Activation(self.activation))

        model.add(Conv2D(32, (3, 3), use_bias=self.use_bias, padding='same'))
        model.add(Activation(self.activation))
        model.add(Conv2D(32, (3, 3), use_bias=self.use_bias, padding='same', strides=(2, 2)))
        model.add(Activation(self.activation))

        model.add(Conv2D(64, (3, 3), use_bias=self.use_bias, padding='same'))
        model.add(Activation(self.activation))
        model.add(Conv2D(64, (3, 3), use_bias=self.use_bias, padding='same', strides=(2, 2)))
        model.add(Activation(self.activation))

        model.add(Conv2D(10, (1, 1), use_bias=self.use_bias, padding='valid'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        print(model.summary())
        return model

    def train(self):
        batch_size = 32
        nb_classes = 10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

        datagen.fit(x_train)
        filepath = self.id_str + '_weights.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='max')

        callbacks_list = [checkpoint]
        if self.use_weight_shaper:
            callbacks_list.append(WeightShaper(self.weight_shaper_max_output, self.weight_shaper_cooldown))

        model = self.keras_model()
        history_callback = model.fit_generator(datagen.flow(x_train, y_train,
                                                            batch_size=batch_size),
                                               steps_per_epoch=x_train.shape[0] // batch_size,
                                               epochs=self.epochs, validation_data=(x_test, y_test),
                                               callbacks=callbacks_list,
                                               verbose=1)

        with open(self.id_str + '_hist.pkl', 'wb') as f:
            pickle.dump(history_callback.history, f)


if __name__ == '__main__':
    fire.Fire(Cifar10Simple)
