import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import cv2
from glob import glob
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import sys
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
#from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from tensorflow.keras.optimizers import SGD,Adam
from keras.optimizers import *
from keras.layers import *        
from keras.applications.vgg16 import VGG16
import keras
import glob
from tensorflow.keras.layers.experimental import preprocessing



import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x
def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x

def encoder1(inputs):
    skip_connections = []

    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)

    output = model.get_layer("block5_conv4").output
    return output, skip_connections
class tich(tf.keras.layers.Layer):
    def __init__(self,b,c,**kwargs,):
        super(tich, self).__init__(**kwargs)
        #self.w = tf.random_normal_initializer()
        self.a = K.variable(value=b, dtype="float32",name = str(c))
        #tf.Variable(b,trainable=True,)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'a': self.a.numpy(),
        })
        return config
    def call(self, inputs):
        return self.a*inputs


def decoder1(inputs, batch,j=1):
    num_filters = [256, 128, 64, 32]
    #skip_connections.reverse()
    x = inputs
    #a = []
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        bien = tich(0.5,j)(batch[i][0]) + tich(0.3,j+1)(batch[i][1]) + tich(0.2,j+2)(batch[i][2])
        bien = BatchNormalization()(bien)
        bien = Activation("relu")(bien)
        #2 = Conv2D(x.shape[-1], 1, dilation_rate=1, padding="same", use_bias=False)(skip_connection[i])
        #b2 = BatchNormalization()(b2)
        #b2 = Activation("relu")(b2)
        b2 = tich(1,j+3)(bien)
        b2 = BatchNormalization()(b2)
        b2 = Activation("relu")(b2)
        bien = Concatenate()([bien,bien*b2])
        x = Concatenate()([x, bien])
        x = conv_block(x, f)
        j=j+4
    return x
def decoder11(inputs, batch,j=25):
    num_filters = [256, 128, 64, 32]
    #skip_connections.reverse()
    x = inputs
    #a = []
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        bien = tich(0.3,j)(batch[i][0]) + tich(0.5,j+1)(batch[i][1]) + tich(0.2,j+2)(batch[i][2])
        bien = BatchNormalization()(bien)
        bien = Activation("relu")(bien)
        #2 = Conv2D(x.shape[-1], 1, dilation_rate=1, padding="same", use_bias=False)(skip_connection[i])
        #b2 = BatchNormalization()(b2)
        #b2 = Activation("relu")(b2)
        b2 = tich(1,j+3)(bien)
        b2 = BatchNormalization()(b2)
        b2 = Activation("relu")(b2)
        bien = Concatenate()([bien,bien*b2])
        x = Concatenate()([x, bien])
        x = conv_block(x, f)
        j=j+4
    return x
def decoder12(inputs, batch,j=50):
    num_filters = [256, 128, 64, 32]
    #skip_connections.reverse()
    x = inputs
    #a = []
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        bien = tich(0.3,j)(batch[i][0]) + tich(0.2,j+1)(batch[i][1]) + tich(0.5,j+2)(batch[i][2])
        bien = BatchNormalization()(bien)
        bien = Activation("relu")(bien)
        #2 = Conv2D(x.shape[-1], 1, dilation_rate=1, padding="same", use_bias=False)(skip_connection[i])
        #b2 = BatchNormalization()(b2)
        #b2 = Activation("relu")(b2)
        b2 = tich(1,j+3)(bien)
        b2 = BatchNormalization()(b2)
        b2 = Activation("relu")(b2)
        bien = Concatenate()([bien,bien*b2])
        x = Concatenate()([x, bien])
        x = conv_block(x, f)
        j=j+4
    return x

def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x

def Upsample(tensor, size):
    """Bilinear upsampling"""
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)

def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def build_model(shape):
    inputs = Input(shape)
    x, skip = encoder1(inputs)


    x1 = ASPP(x,16)

    batch1 = [Conv2D(64,(1,1),padding="same")(skip[3]),Conv2D(64,(3,3),padding="same")(skip[3]),Conv2D(64,(5,5),padding="same")(skip[3])]
    batch2 = [Conv2D(256,(1,1),padding="same")(skip[2]),Conv2D(256,(3,3),padding="same")(skip[2]),Conv2D(256,(5,5),padding="same")(skip[2])]
    batch3 = [Conv2D(128,(1,1),padding="same")(skip[1]),Conv2D(128,(3,3),padding="same")(skip[1]),Conv2D(128,(5,5),padding="same")(skip[1])]
    batch4 = [Conv2D(64,(1,1),padding="same")(skip[0]),Conv2D(64,(3,3),padding="same")(skip[0]),Conv2D(64,(5,5),padding="same")(skip[0])]
    batch = [batch1,batch2,batch3,batch4]
    x1 = decoder1(x1, batch)
    outputs1 = output_block(x1)
    x2 = ASPP(x,32)
    x2 = decoder11(x2, batch)
    outputs2 = output_block(x2)
    x3 = ASPP(x,64)
    x3 = decoder12(x3, batch)
    outputs3 = output_block(x3)
    
    outputs = Concatenate()([ASPP(outputs1,1),ASPP(outputs2,1),ASPP(outputs3,1)])
    outputs = Conv2D(1,(1,1),activation='sigmoid')(outputs)
    model = Model(inputs, outputs)
    return model

if __name__ == '__main':
   model = build_model((256,256, 3))
   model(np.random.rand(1,256,256,3)).shape
