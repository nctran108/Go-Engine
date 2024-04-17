import sys
import os
sys.path.append(os.getcwd())

import argparse
from go.data.parallel_processor import GoDataProcessor
from go.encoders import ZeroEncoder
from go.agent import ZeroAgent
from go.RL import ZeroExperienceCollector, combine_zero_experience
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, Input
from go.gotypes import Player
from go.goboard import GameState
from go.score import compute_game_result

from keras.models import Model
import h5py
import numpy as np

def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = compute_game_result(game)
    print(game_result)
    # Give the reward to the right agent.
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)

def main(args):
    board_size = 13
    encoder = ZeroEncoder(board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input

    for i in range(4):
        pb = Conv2D(64, (3,3),
                    padding='same',
                    data_format='channels_first',
                    activation='relu')(pb)

    policy_conv = Conv2D(2, (1,1),
                        data_format='channels_first',
                        activation='relu')(pb)

    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(encoder.num_moves(), activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1,1),
                        data_format='channels_first',
                        activation='relu')(pb)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(inputs=[board_input],
                outputs=[policy_output,value_output])

    black_agent = ZeroAgent(model, encoder, rounds_per_move=1600, c=2.0)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=1600, c=2.0)

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()

    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    for i in range(5):
        simulate_game(board_size, black_agent, c1, white_agent, c2)

    exp = combine_zero_experience([c1, c2])

    black_agent.train(exp, 0.01, 2048)

    with h5py.File('bots/13x13_zero_1600_rounds_5_games.h5', 'w') as agent_outf:
        exp.serialize(agent_outf)

if __name__ == "__main__":
    main()