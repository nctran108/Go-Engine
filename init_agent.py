import argparse

import h5py

from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from networks import large
from go import agent
from go.encoders import get_encoder_by_name


def main():
    board_size = 19
    output_file = './large_model_sevenplane_encoder_with_SGD.h5'

    encoder = get_encoder_by_name('sevenplane', board_size=board_size)
    model = Sequential()
    for layer in large.layers(encoder.shape()):
        model.add(layer)
    model.add(Dense(encoder.num_points()))
    model.add(Activation('softmax'))
    opt = SGD(learning_rate=0.02)
    model.compile(loss=agent.policy_gradient_loss, optimizer=opt)

    new_agent = agent.PolicyAgent(model, encoder)

    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)

if __name__ == '__main__':
    main()