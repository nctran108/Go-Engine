
import sys
sys.path.append('d:\\study\\Go-Engine')
sys.path.append('d:\study\Go-Engine\go\RL')
from go.goboard import GameState, Player
from go import score
from go import RL
from go import agent
from go.kerasutil import load_model_from_hdf5_group
from go.encoders.sevenplane import SevenPlaneEncoder

from collections import namedtuple
import h5py
from keras import Sequential
from networks import large
from keras.layers import Dense, Dropout
BOARD_SIZE = 19

class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass

def simulate_game(black_player, white_player):
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    moves = []
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    game_result = score.compute_game_result(game)
    return GameRecord(moves=moves,
                      winner=game_result.winner,
                      margin=game_result.winning_margin)

def experience_simulation(num_games, agent1, agent2):
    collector1 = RL.ExperienceCollector()
    collector2 = RL.ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)
        if game_record.winner == Player.black:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=1)
    
    return RL.combine_experience([collector1,collector2])

def main():
    agent1 = agent.load_policy_agent(h5py.File('large_model_sevenplane_encoder_with_SGD.h5', 'r'))
    agent2 = agent.load_policy_agent(h5py.File('large_model_sevenplane_encoder_with_SGD.h5', 'r'))
    experience = experience_simulation(1000,agent1=agent1,agent2=agent2)
    with h5py.File('policy_bot_1000_games.h5', 'w') as experience_outf:
        experience.serialize(experience_outf)

if __name__ == '__main__':
    main()