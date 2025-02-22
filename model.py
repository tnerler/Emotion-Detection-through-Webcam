import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from keras.models import Sequential

INPUT_SHAPE = (48, 48, 3)

def create_model(input_shape=INPUT_SHAPE) : 

    model = Sequential()

    # 1. Block

    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape= input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2. Block

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3. Block

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4. Block
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 5. Block
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    

    model.add(Dense(7, activation='softmax'))

    return model

