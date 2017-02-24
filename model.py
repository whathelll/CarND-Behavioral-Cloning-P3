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
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

class MyModel():
    'this builds the model'
    def __init__(self, batch_size=256, nb_epoch=10):
        self.image_size_x=64
        self.image_size_y=64
        self.batch_size=batch_size
        self.nb_epoch=nb_epoch
        
        model = self.custom_VGG()
        model.summary()
        
        self.model = model

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

    def custom_VGG(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(self.image_size_x, self.image_size_y, 3)))
        model.add(Convolution2D(32, 1, 1, border_mode='same'))
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(Convolution2D(128, 3, 3, border_mode='same'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(AveragePooling2D(pool_size=(2,2), border_mode='same'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='elu', name='dense1'))
        model.add(Dense(1024, activation='elu', name='dense2'))
        model.add(Dense(512, activation='elu', name='dense3'))
        model.add(Dense(256, activation='elu', name='dense4'))
        model.add(Dense(128, activation='elu', name='dense5'))
        model.add(Dense(64, activation='elu', name='dense6'))
        model.add(Dense(32, activation='elu', name='dense7'))
        model.add(Dense(1))
        adam = Adam(lr=0.00005)
        model.compile(adam, 'mse')
        return model
    
    
    def nvidia(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(self.image_size_x, self.image_size_y,3)))
        model.add(Convolution2D(24, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(36, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
        #model.add(Dropout(0.5))
        model.add(ELU())
        model.add(Flatten())
        model.add(Dense(1164))
        model.add(ELU())
        model.add(Dense(100))
        model.add(ELU())
        model.add(Dense(50))
        model.add(ELU())
        model.add(Dense(10))
        model.add(ELU())
        model.add(Dense(1))
        adam = Adam(lr=0.0001)
        model.compile(adam, 'mse')
        return model

    
    def get_images(self, df, camera='center'):
        result = []
        for path in df[camera]:
            img = mpimg.imread(path)
            img = misc.imresize(img, (self.image_size_x, self.image_size_y))
            #img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
            result.append(img)
        return np.stack(result)

    def load_data(self):
        df = pd.read_csv('./data/driving_log.csv')
        df = df.replace(to_replace="( )?IMG", value='./data/IMG', regex=True)
        
        # sample 500 straight driving images
        df_zero = df[(df['steering'] > -0.01) | (df['steering'] < 0.01)].reset_index(drop=True)
        df_zero = df_zero.sample(n=500)
        x_zero = self.get_images(df_zero)
        y_zero = df_zero.as_matrix(columns=['steering'])
        
        df = df[(df['steering'] <= -0.01) | (df['steering'] >= 0.01)].reset_index(drop=True)
        x = self.get_images(df)
        y = df.as_matrix(columns=['steering'])
        
        # let's add in left and right cameras too
        steering_adjustment = 0.1
        x_left = self.get_images(df, camera='left')
        y_left = y + steering_adjustment
        x_right = self.get_images(df, camera='right')
        y_right = y - steering_adjustment
        
        x = np.append(x, x_left, axis=0)
        y = np.append(y, y_left, axis=0)
        x = np.append(x, x_right, axis=0)
        y = np.append(y, y_right, axis=0)
        x = np.append(x, x_zero, axis=0)
        y = np.append(y, y_zero, axis=0)

        # flip the images and steering and then add to the array
        x_flip = [np.fliplr(a) for a in x]
        y_flip = -y
        x = np.append(x, x_flip, axis=0)
        y = np.append(y, y_flip, axis=0)
        
        # shuffle
        x, y = shuffle(x, y)
        
        self.x = x
        self.y = y

    def trans_image(self, image, steer, translation_range):
        # Translation
        x_translation = translation_range*np.random.uniform()-translation_range/2
        steer_ang = steer + x_translation/translation_range*2*.2
        y_translation = 40*np.random.uniform()-40/2

        translation_matrix = np.float32([[1,0,x_translation],[0,1,y_translation]])
        image_tr = cv.warpAffine(image,translation_matrix,(image.shape[1],image.shape[0]))

        return image_tr,steer_ang
        
    def crop_resize(self, img):
        img = img[50:130,0:320]
        img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
        return cv.resize(img, (self.image_size_x, self.image_size_y), interpolation=cv.INTER_AREA)

    def preprocess_img(self, row):
        img = None
        steering = None
        camera = ['left', 'center', 'right']
        adjustment_value = 0.1
        steering_adjustment = [adjustment_value, 0, -adjustment_value]
        camera_selection = random.randint(0, 2)

        steering = row['steering'] + steering_adjustment[camera_selection]

        img = mpimg.imread(row[camera[camera_selection]])
        #img, steering = self.trans_image(img,steering,100)
        img = self.crop_resize(img)

        #random flip
        flip = random.randint(0, 1)
        if flip == 1: 
            img = np.fliplr(img)
            steering = -steering

        return img, steering
    
    def generator(self, df, batch_size=256):
        while 1: # Loop forever so the generator never terminates
            x_sample = []
            y_sample = []
            pr_threshold = 1
            for i in range(batch_size):
                index = np.random.randint(len(df))
                row = df.iloc[index]
                image, steering = self.preprocess_img(row)
#                    keep_pr = 0
#                    while keep_pr == 0:
#                        image, steering = self.preprocess_img(row)
#                        pr_unif = np.random
#                        if abs(steering)<.1:
#                            pr_val = np.random.uniform()
#                            if pr_val>pr_threshold:
#                                keep_pr = 1
#                        else:
#                            keep_pr = 1
#                    pr_threshold = 0.99 * pr_threshold
                x_sample.append(image)
                y_sample.append(steering)
            yield np.stack(x_sample), np.asarray(y_sample).reshape(batch_size, 1)
#                    pr_threshold = 0.90 * pr_threshold

    def validation_generator(self, batch_size=256):
        df = pd.read_csv('./data/driving_log.csv')
        df = df.replace(to_replace="( )?IMG", value='./data/IMG', regex=True)
        df = df[(df['steering'] <= -0.01) | (df['steering'] >= 0.01)].reset_index(drop=True)
        while 1: # Loop forever so the generator never terminates
            x_sample = []
            y_sample = []

            size = len(df)
            for i in range(batch_size):
                index = np.random.randint(size)
                row = df.iloc[index]
                image = mpimg.imread(df['center'][index])
                image = self.crop_resize(image)
                steering = df['steering'][index]

                x_sample.append(image)
                y_sample.append(steering)
            yield np.stack(x_sample), np.asarray(y_sample).reshape(batch_size, 1)

    def load_generators(self):
        df = pd.read_csv('./data/driving_log.csv')
        df = df.replace(to_replace="( )?IMG", value='./data/IMG', regex=True)
        df = df[(df['steering'] <= -0.01) | (df['steering'] >= 0.01)].reset_index(drop=True)
        self.train_generator = self.generator(df)
        self.validation_generator = self.validation_generator()
        
    def train(self):
        self.model.fit(self.x, self.y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, validation_split=0.2)
        
    def train_with_generator(self):
        self.model.fit_generator(self.train_generator, 
                                 validation_data=self.validation_generator, 
                                 nb_val_samples=self.batch_size*4, 
                                 samples_per_epoch= self.batch_size*40, nb_epoch=self.nb_epoch)

    def evaluate(self):
        result = self.model.evaluate(self.self_xtest, self.self_ytest, batch_size=self.batch_size)
        print(self.model.metrics_names)
        print(result)
        
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
    #model.load_data()
    model.load_generators()
    
    print('training model')
    #model.train()
    model.train_with_generator()

    print('evaluating')
    #model.evaluate()
    
    print('saving')
    model.save()
    print('finished')
    
    
    
