# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 22:38:12 2020
@author: aguemes
"""


import os
import re
import math
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 

print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)


def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.BatchNormalization(momentum = 0.5)(model)
    model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(model)
    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.BatchNormalization(momentum = 0.5)(model)
        
    model = layers.Add()([gen, model])
    
    return model
    
    
def up_sampling_block(model, kernal_size, filters, strides):

    model = layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.UpSampling2D(size = 2, data_format='channels_first')(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    model = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_first')(model)
    model = layers.BatchNormalization(momentum = 0.5)(model)
    model = layers.LeakyReLU(alpha = 0.2)(model)
    
    return model


class Generator(object):


    def __init__(self, input_shape=(3,96,96), upsamples=1):
        
        self.input_shape = input_shape
        self.upsamples = upsamples


    def generator(self):
        
	    gen_input = keras.Input(shape = self.input_shape)
	    model = layers.Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same", data_format='channels_first')(gen_input)
	    model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[2,3])(model)  
	    gen_model = model
        
	    for index in range(16):

	        model = res_block_gen(model, 3, 64, 1)
	    
	    model = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_first')(model)
	    model = layers.BatchNormalization(momentum = 0.5)(model)
	    model = layers.Add()([gen_model, model])
	    
	    for index in range(self.upsamples):

	        model = up_sampling_block(model, 3, 256, 1)
	    
	    model = layers.Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same", data_format='channels_first')(model)
	    model = layers.Activation('tanh')(model)
	    generator_model = keras.Model(inputs = gen_input, outputs = model)
        
	    return generator_model


class Discriminator(object):


    def __init__(self, output_shape=(3,192,192)):
        
        self.output_shape = output_shape
    

    def discriminator(self):
        
        dis_input = keras.Input(shape = self.output_shape)
        model = layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_first')(dis_input)
        model = layers.LeakyReLU(alpha = 0.2)(model)
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        model = layers.Flatten()(model)
        model = layers.Dense(1024)(model)
        model = layers.LeakyReLU(alpha = 0.2)(model)
        model = layers.Dense(1)(model)
        model = layers.Activation('sigmoid')(model) 
        discriminator_model = keras.Model(inputs = dis_input, outputs = model)
        
        return discriminator_model