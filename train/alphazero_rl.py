import sys
import os
sys.path.append(os.getcwd())

import argparse
from go.data.parallel_processor import GoDataProcessor
from go.encoders import ZeroEncoder
from go.agent import ZeroAgent, load_zero_agent
from go.RL import ZeroExperienceCollector, combine_zero_experience
from go.gotypes import Player
from go.goboard import GameState
from go.score import compute_game_result
from networks import alphaZero

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

def main():
    board_size = 13
    encoder = ZeroEncoder(board_size)

    model = alphaZero.model(encoder)
    black_agent = ZeroAgent(model, encoder, rounds_per_move=1600, c=2.0)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=1600, c=2.0)

    #print(os.getcwd())
    #black_agent = load_zero_agent(h5py.File('bots/13x13_zero_1600_rounds_5_games.h5'))
    #white_agent = load_zero_agent(h5py.File('bots/13x13_zero_1600_rounds_5_games.h5'))

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()

    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    num_games = 10
    for i in range(num_games):
        simulate_game(board_size, black_agent, c1, white_agent, c2)

    exp = combine_zero_experience([c1, c2])

    black_agent.train(exp, 0.01, 2048)

    with h5py.File('bots/13x13_zero_1600_rounds_10_games.h5', 'w') as agent_outf:
        black_agent.serialize(agent_outf)

if __name__ == "__main__":
    main()