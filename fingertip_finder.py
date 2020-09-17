import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, UpSampling2D, Activation
from tensorflow.keras.applications import VGG16


def model():
    model = VGG16(include_top=False, input_shape=(128, 128, 3))
    x = model.output

    y = x
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    p = Dense(5, activation='sigmoid', name='probabilistic_output')(x)

    y = UpSampling2D((3, 3))(y)
    y = Activation('relu')(y)
    y = Conv2D(1, (3, 3), activation='linear')(y)
    position = Reshape(target_shape=(10, 10), name='positional_output')(y)
    model = Model(model.input, outputs=[p, position])
    return model


class Fingertips:
    def __init__(self, weights):
        self.model = model()
        self.model.load_weights(weights)

    @staticmethod
    def class_finder(prob):
        c = ''
        classes = [0, 1, 2, 3, 4, 5, 6, 7]

        if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
            c = classes[0]
        elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
            c = classes[1]
        elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
            c = classes[2]
        elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
            c = classes[3]
        elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
            c = classes[4]
        elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
            c = classes[5]
        elif np.array_equal(prob, np.array([1, 1, 0, 0, 1])):
            c = classes[6]
        elif np.array_equal(prob, np.array([1, 1, 0, 0, 0])):
            c = classes[7]
        return c

    def classify(self, im):
        im = np.asarray(im)
        im = cv2.resize(im, (128, 128))
        im = im.astype('float32')
        im = im / 255.0
        im = np.expand_dims(im, axis=0)
        p, pos = self.model.predict(im)
        p = p[0]
        pos = pos[0]
        return p, pos
