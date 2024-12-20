from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D

def layers(input_shape):
    return [
        ZeroPadding2D((3,3), input_shape=input_shape, data_format='channels_first'),
        Conv2D(64,kernel_size=(7,7), padding='valid', data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2,2), data_format='channels_first'),
        Conv2D(64,kernel_size=(5,5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2,2), data_format='channels_first'),
        Conv2D(64,kernel_size=(5,5), data_format='channels_first'),
        Activation('relu'),

        Dropout(rate=0.5),

        ZeroPadding2D((2,2), data_format='channels_first'),
        Conv2D(48,kernel_size=(5,5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2,2), data_format='channels_first'),
        Conv2D(48,kernel_size=(5,5), data_format='channels_first'),
        Activation('relu'),


        ZeroPadding2D((2,2), data_format='channels_first'),
        Conv2D(32,kernel_size=(5,5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D((2,2), data_format='channels_first'),
        Conv2D(32,kernel_size=(5,5), data_format='channels_first'),
        Activation('relu'),

        Flatten(),
        Dense(512), 
        Activation('relu'),
        Dense(1024), 
        Activation('relu'),
        Dense(512), 
        Activation('relu'),
    ]