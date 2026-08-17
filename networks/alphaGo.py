from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

def alphaGoModel(input_shape, is_policy_net=False,
                  num_filters=192,
                  first_kernel_size=(5, 5),
                  other_kernel_size=(3, 3)):
    model = Sequential()
    model.add(Conv2D(num_filters, first_kernel_size,
                      padding='same', data_format='channels_last',
                        activation='relu', input_shape=input_shape))
    for _ in range(2, 12):
        model.add(Conv2D(num_filters, other_kernel_size,
                          padding='same', data_format='channels_last',
                           activation='relu'))

    if is_policy_net:
        model.add(Conv2D(num_filters, (1,1),
                          padding='same', data_format='channels_last',
                          activation='softmax'))
        model.add(Flatten())
    else:
        model.add(Conv2D(num_filters, other_kernel_size,
                          padding='same', data_format='channels_last',
                          activation='relu'))
        model.add(Conv2D(1, (1,1),
                          padding='same', data_format='channels_last',
                          activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='tanh'))
    return model