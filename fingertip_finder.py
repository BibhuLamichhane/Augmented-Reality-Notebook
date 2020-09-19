import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, UpSampling2D, Activation
from tensorflow.keras.applications import VGG16


def model():
    m = VGG16(include_top=False, input_shape=(128, 128, 3))
    x = m.output

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
    m = Model(m.input, outputs=[p, position])
    return m


class Fingertips:
    def __init__(self, weights):
        self.model = model()
        self.model.load_weights(weights)

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
