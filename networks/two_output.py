from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input

def layers(encoder):
    board_input = Input(shape=encoder.shape(), name='board_input')

    # all convolution layers to organizes the pieces on the board into logical groups
    # can add as many conv layer as I want
    conv1 = Conv2D(64, (3,3),
                   padding='same',
                   activation='relu')(board_input)
    
    conv2 = Conv2D(64, (3,3),
                   padding='same',
                   activation='relu')(conv1)
    
    conv3 = Conv2D(64, (3,3),
                   padding='same',
                   activation='relu')(conv2)
    
    # flat the output for dense layer
    flat = Flatten()(conv3)
    # Input layer
    processed_board = Dense(512)(flat)

    # hidden layer
    policy_hidden_layer = Dense(512, activation='relu')(processed_board)

    # policy output which is probability distribution over moves (policy or actor)
    policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)

    # The value layer for expected reward output
    value_hidden_layer = Dense(512, activation='relu')(policy_hidden_layer)
    # use tanh for result between -1 and 1
    value_output = Dense(1, activation='tanh')(value_hidden_layer)

    return Model(inputs=board_input,
                 outputs=[policy_output, value_output])