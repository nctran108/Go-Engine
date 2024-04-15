from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D
from tqdm import tqdm

def alphago_model(input_shape, is_policy_net=False,
                  num_filters=192,
                  first_kernel_size=5,
                  other_kernel_size=3):
    model = Sequential()
    # input 19x19 layer, and output is 192 which is 19x19
    # with kernel size is 5 and keep the padding with same size for other layers
    model.add(Conv2D(num_filters,first_kernel_size,input_shape=input_shape,padding='same',data_format='channels_first',activation='relu'))

    # the first 12 layers of policy and value network are identical
    # the output is 19x19 = 192
    print("Adding 11 Convrtional layers.........")
    for i in tqdm(range(2, 12)):
        model.add(Conv2D(num_filters, other_kernel_size,
                         padding='same',
                         data_format='channels_first',
                         activation='relu'))
        if i == 5:
            Dropout(rate=0.5)
    
    Dropout(rate=0.5)
    if is_policy_net:
        # this is for policy network
        # output = 1
        model.add(Conv2D(filters=1, kernel_size=1, padding='same',
                         data_format='channels_first', activation='softmax'))
        model.add(Flatten())
        print("added policy layers...")
        return model
    
    # if not policy then
    # add value network layers
    model.add(Conv2D(num_filters, other_kernel_size, padding='same',
                     data_format='channels_first', activation='relu'))
     
    model.add(Conv2D(filters=1, kernel_size=1, padding='same',
                         data_format='channels_first', activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    print("added value layers...")
    return model

