import tensorflow as tf
from tensorflow import keras
from keras import optimizers, layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import  VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import regularizers
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, LSTM, Conv3D, MaxPool3D, Conv1D, MaxPool1D, concatenate
from keras import initializers
from keras import optimizers
from keras import losses
import numpy as np
import os

def build_CNN_model3D():
    # Define the model architecture
    model = keras.models.Sequential()
    model.add(keras.models.Conv3D(filters=2, kernel_size=3, activation='relu',
                     input_shape=[128, 128, 22, 1]))
    model.add(keras.models.MaxPooling3D(pool_size=3))
    model.add(keras.models.Conv3D(filters=4, kernel_size=3, activation='relu'))
    model.add(keras.models.Conv3D(filters=4, kernel_size=3, activation='relu'))
    model.add(keras.models.Flatten())
    model.add(keras.models.Dense(1024, activation='relu'))
    model.add(keras.models.Dense(512, activation='relu'))
    model.add(keras.models.Dense(1))

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy', 'mean_absolute_error'])
    return model


def build_CNN_model2D():
    # Define the model architecture
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[128, 128, 22],data_format="channels_last"))
    model.add(keras.layers.MaxPooling2D(pool_size=3))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=3))
    model.add(keras.layers.Conv2D(filters=34, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=3))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    return model

def vgg16():
    model3d = keras.models.Sequential()
    model3d.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(128, 128, 22),data_format="channels_last",activation='relu', 
                       kernel_initializer=initializers.glorot_normal()))
    model3d.add(Conv2D(filters=10, kernel_size=(3, 3),activation='relu', kernel_initializer=initializers.glorot_normal()))
    model3d.add(Conv2D(filters=3, kernel_size=(3, 3),activation='relu', kernel_initializer=initializers.glorot_normal()))


    
    ## Loading VGG16 model
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=[122, 122, 3])
    base_model.trainable = False ## Not trainable weights
    
    flatten_layer = keras.layers.Flatten()
    dense_layer_1 = keras.layers.Dense(50, activation='relu')
    dense_layer_2 = keras.layers.Dense(20, activation='relu')
    prediction_layer = keras.layers.Dense(2, activation='softmax')
    
    model = keras.models.Sequential([
        model3d,
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    return model

def vgg19():
    model3d = keras.models.Sequential()
    model3d.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(128, 128, 22),data_format="channels_last",activation='relu', 
                       kernel_initializer=initializers.glorot_normal()))
    model3d.add(Conv2D(filters=10, kernel_size=(3, 3),activation='relu', kernel_initializer=initializers.glorot_normal()))
    model3d.add(Conv2D(filters=3, kernel_size=(3, 3),activation='relu', kernel_initializer=initializers.glorot_normal()))


    
    ## Loading VGG16 model
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=[122, 122, 3])
    base_model.trainable = False ## Not trainable weights
    
    flatten_layer = keras.layers.Flatten()
    dense_layer_1 = keras.layers.Dense(50, activation='relu')
    dense_layer_2 = keras.layers.Dense(20, activation='relu')
    prediction_layer = keras.layers.Dense(2, activation='softmax')
    
    model = keras.models.Sequential([
        model3d,
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=["accuracy"])
    return model

