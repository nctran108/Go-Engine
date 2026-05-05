import sys
import os
import signal
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

game_played = 0
CONTROL_C = False

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

def signal_handler(sig, frame):
    global CONTROL_C
    print('CTRL-C was pressed')
    CONTROL_C = True

def main():
    global CONTROL_C
    signal.signal(signal.SIGINT, signal_handler)
    board_size = 19

    black_agent = load_zero_agent('bots/19x19_zero_1600_rounds_first_games.weights.h5', json_file=True)
    white_agent = load_zero_agent('bots/19x19_zero_1600_rounds_first_games.weights.h5', json_file=True)

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    #encoder = ZeroEncoder(board_size)

    #model = alphaZero.model(encoder)
    #black_agent = ZeroAgent(model, encoder, rounds_per_move=1600, c=2.0)
    #white_agent = ZeroAgent(model, encoder, rounds_per_move=1600, c=2.0)

    #print(os.getcwd())
    

    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    num_games = 10
    
    for i in range(num_games):
        simulate_game(board_size, black_agent, c1, white_agent, c2)
        game_played += 1
        if CONTROL_C:
            break

    exp = combine_zero_experience([c1, c2])

    black_agent.train(exp, 0.01, 2048)

    black_agent.serialize(f'bots/19x19_zero_1600_rounds_{game_played}_games.weights.h5', json_file=True)     

if __name__ == "__main__":
    main()