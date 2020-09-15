import cv2
import numpy as np
from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation


def conv_batch_norm_relu(x, n_filters, f, padding='same', activation='relu'):
    x = Conv2D(n_filters, f, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def model():
    i = Input(shape=(224, 224, 3))
    x = conv_batch_norm_relu(i, 32, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 64, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 128, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 64, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 128, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 256, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 128, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 256, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 512, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 256, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 256, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (3, 3), padding='same', activation='relu')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_batch_norm_relu(x, 1024, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 1024, (3, 3), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 512, (1, 1), padding='same', activation='relu')
    x = conv_batch_norm_relu(x, 1024, (3, 3), padding='same', activation='relu')
    x = Conv2D(5, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid', name='output')(x)
    return Model(inputs=input, outputs=x)


class Flag:
    def __init__(self):
        self.grid = 7
        self.grid_size = 32
        self.target_size = 224
        self.threshold = 0.5
        self.alpha = 0.5
        self.line_color = (18, 203, 227)
        self.grid_color = (81, 189, 42)
        self.box_color = (235, 26, 158)


class YOLO:
    def __init__(self, weights, threshold):
        self.f = Flag()
        self.model = model()
        self.threshold = threshold
        self.model.load_weights(weights)

    def detect(self, image):
        height, width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.f.target_size, self.f.target_size)) / 255.0
        image = np.expand_dims(image, axis=0)
        yolo_out = self.model.predict(image)
        yolo_out = yolo_out[0]

        grid_pred = yolo_out[:, :, 0]
        i, j = np.squeeze(np.where(grid_pred == np.amax(grid_pred)))

        if grid_pred[i, j] >= self.threshold:
            bbox = yolo_out[i, j, 1:]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            # size conversion
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            return (x1, y1), (x2, y2)
        else:
            return None, None
