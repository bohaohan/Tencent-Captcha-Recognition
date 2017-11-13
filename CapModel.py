#-*- coding: utf-8 -*-
__author__ = 'bohaohan'
import numpy as np

from keras.applications.xception import preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from scipy import misc


class CapModel:

    def __init__(self, img_size=(50, 120)):
        self.img_size = img_size
        self.model = self.get_model()

    def pre_process_test_img(self, path):
        test_x = [misc.imresize(misc.imread(path), self.img_size)]
        return preprocess_input(np.array(test_x).astype(float))

    def get_model(self, pre_train_path="cap1.h5"):
        print "Loading model..."
        input_image = Input(shape=(self.img_size[0], self.img_size[1], 3))
        base_model = MobileNet(input_tensor=input_image, weights=None, include_top=False, pooling='avg')

        predicts = [Dense(26, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

        model = Model(inputs=input_image, outputs=predicts)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        model.load_weights(pre_train_path)
        print "Load model successfully!"
        return model

    def predict(self, path="test_cap.png"):
        result = ""
        test_x = self.pre_process_test_img(path)
        prediction = self.model.predict(test_x)
        prediction = np.array([i.argmax(axis=1) for i in prediction]).T

        for char in prediction[0]:
            result += chr(char + ord('a'))
        return result


if __name__ == '__main__':
    model = CapModel()
    model.predict(path="")