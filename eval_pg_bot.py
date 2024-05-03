import argparse
import datetime
from collections import namedtuple

import h5py

from go.agent import load_policy_agent, load_zero_agent
from go import score
from go.goboard import GameState, Player, Point
from tqdm import tqdm
from go.utils import print_board


BOARD_SIZE = 13

def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        #if next_move.is_pass:
        #    print('%s passes' % name(game.next_player))
        game = game.apply_move(next_move)

    print_board(game.board)
    game_result = score.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)

    args = parser.parse_args()

    agent1 = load_zero_agent(h5py.File(args.agent1))
    agent2 = load_zero_agent(h5py.File(args.agent2))

    wins = 0
    losses = 0
    color1 = Player.black
    for i in tqdm(range(args.num_games)):
        #print('Simulating game %d/%d...' % (i + 1, args.num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            wins += 1
        else:
            losses += 1
        color1 = color1.other
    print('Agent 1 record: %d/%d' % (wins, wins + losses))


if __name__ == '__main__':
    main()