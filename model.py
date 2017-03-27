#!/usr/bin/env python

import argparse
import random

import pandas as pd
import numpy as np
import cv2 as cv
import scipy.misc as misc
import matplotlib.image as mpimg
from sklearn.utils import shuffle


import keras
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Input, ELU
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

class MyModel():
    def __init__(self, batch_size=256, nb_epoch=10):
        self.image_size_x=64
        self.image_size_y=64
        self.batch_size=batch_size
        self.nb_epoch=nb_epoch
        
        model = self.nvidia()
        model.summary()
        
        self.model = model
    """
        taking the VGG16 model from keras and put layers in front and after it
    """
    def VGG(self):
        inp = Input(shape=(self.image_size_x, self.image_size_y, 3))
        preproc = Lambda(lambda x: (x / 255.0) - 0.5)(inp)
        modelvgg = VGG16(include_top=False, weights=None, input_tensor=preproc, input_shape=None)
        layer = Flatten(name='flatten')(modelvgg.layers[-1].output)
        layer = Dense(2048, activation='elu', name='dense1')(layer)
        layer = Dense(2048, activation='elu', name='dense2')(layer)
        layer = Dense(512, activation='elu', name='dense2.1')(layer)
        layer = Dense(128, activation='elu', name='dense3')(layer)
        layer = Dense(32, activation='elu', name='dense4')(layer)
        prediction = Dense(1)(layer)
        model = Model(modelvgg.input, prediction)
        adam = Adam(lr=0.0001)
        model.compile(adam, 'mse')
        return model
    """
        building a modified version of VGG which is a lighter weight CNN with the ability to adjust everything
    """
    def custom_VGG(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(self.image_size_x, self.image_size_y, 3)))
        model.add(Convolution2D(32, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        adam = Adam(lr=0.00001)
        model.compile(adam, 'mse')
        return model
    
    """
        Build the nvidia model which I added 2 additional convolution layers to test'
    """
    def nvidia(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(self.image_size_x, self.image_size_y,3)))
        model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(36, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(1164))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        adam = Adam(lr=0.0001)
        model.compile(adam, 'mse')
        return model

    """
        Image manipulation, including cropping, adjusting brightness, RGB2YUV and resize
    """
    def crop_resize(self, img):
        # crop the top and bottom pixels
        img = img[50:130,0:320]

        # randomly adjust brightness 
        # img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        # brightness = 0.25 + np.random.uniform()
        # img[:, :, 2] = img[:, :, 2] * brightness
        # img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        
        #convert to YUV and then resize
        img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
        return cv.resize(img, (self.image_size_x, self.image_size_y), interpolation=cv.INTER_AREA)

    """
        function used to load an image and do some preprocessing/augmentation
        preprocessing include:
            - randomly use left/right/center images
            - call the crop_resize function
            - randomly flip the image
    """
    def preprocess_img(self, row):
        img = None
        steering = None
        camera = ['left', 'center', 'right']
        adjustment_value = 0.05
        steering_adjustment = [adjustment_value, 0, -adjustment_value]
        camera_selection = random.randint(0, 2)

        steering = row['steering'] + steering_adjustment[camera_selection]

        img = mpimg.imread(row[camera[camera_selection]])
        img = self.crop_resize(img)

        #random flip
        flip = random.randint(0, 1)
        if flip == 1: 
            img = np.fliplr(img)
            steering = -steering

        return img, steering

    """
        Training generator: used to generate training data with random sampling then preprocess the image
    """
    def generator(self, df, batch_size=256):
        while 1: # Loop forever so the generator never terminates
            x_sample = []
            y_sample = []
            for i in range(batch_size):
                row = df.sample(n=1).iloc[0]
                image, steering = self.preprocess_img(row)
                x_sample.append(image)
                y_sample.append(steering)
            yield np.stack(x_sample), np.asarray(y_sample).reshape(batch_size, 1)

    """
        Validation generator: produces images similar to the actual data
    """
    def validation_generator(self, batch_size=256):
        df = pd.read_csv('./data/driving_log.csv')
        df = df.replace(to_replace="( )?IMG", value='./data/IMG', regex=True)
        df = df[(df['steering'] <= -0.01) | (df['steering'] >= 0.01)].reset_index(drop=True)
        while 1: # Loop forever so the generator never terminates
            x_sample = []
            y_sample = []

            size = len(df)
            for i in range(batch_size):
                index = np.random.randint(len(df))
                image = mpimg.imread(df['center'][index])
                image = self.crop_resize(image)
                steering = df['steering'][index]

                x_sample.append(image)
                y_sample.append(steering)
            yield np.stack(x_sample), np.asarray(y_sample).reshape(batch_size, 1)

    """
        initialize generators
    """
    def load_generators(self):
        df = pd.read_csv('./data/driving_log.csv')
        df = df.replace(to_replace="( )?IMG", value='./data/IMG', regex=True)
        df = df[(df['steering'] <= -0.01) | (df['steering'] >= 0.01)].reset_index(drop=True)
        self.train_generator = self.generator(df)
        self.validation_generator = self.validation_generator()
        
    """
        Train
    """
    def train_with_generator(self):
        self.model.fit_generator(self.train_generator, 
                                 validation_data=self.validation_generator, 
                                 validation_steps=8,
                                 steps_per_epoch=40, epochs=self.nb_epoch)
        
    """
        save the model file
    """
    def save(self):
        self.model.save('model.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Building')
    parser.add_argument('batch_size', type=int, help='The batch size')
    parser.add_argument('nb_epoch', type=int, help='The number of epochs')
    args = parser.parse_args()
    
    print('initializing model')
    model = MyModel(args.batch_size, args.nb_epoch)
    
    print('loading data')
    model.load_generators()
    
    print('training model')
    model.train_with_generator()
    
    print('saving')
    model.save()
    print('finished')
    
    
    
