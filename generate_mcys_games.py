import argparse
import numpy as np

from go.encoders import get_encoder_by_name
from go import goboard as goboard
from AlphaZero import monte_carlo_tree_search as mcts
from go.utils import print_board, print_move

def generate_game(board_size, rounds, max_moves, temerature):
    # In boards store encoded board state. moves is for encoded moves
    boards, moves = [], []
    # Initialize a OnePlaneEncoder by name with given board size
    encoder = get_encoder_by_name('oneplane', board_size)
    # A new game of size board_size is instantiated
    game = goboard.GameState.new_game(board_size)
    # A Monte Carlo tree-search agent with specified number of rounds and temperature will serve as the bot.
    bot = mcts.MCTSAgent(rounds, temerature)

    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        # the next move is selected by the bot
        move = bot.select_move(game)

        # the encoded board situation is appened to boards
        if move.is_play:
            boards.append(encoder.encode(game))
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            # the one-hot-encoded next move is appended to moves
            moves.append(move_one_hot)

        print_move(game.next_player, move)
        # Afterward, the bot move is appied to the board.
        game = game.apply_move(move)
        num_moves += 1
        # continou with the next move, unless the max number of moves has been reached
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', '-b', type=int, default=9)
    parser.add_argument('--rounds', '-r', type=int, default=1000)
    parser.add_argument('--temperature', '-t', type=float, default=0.8)
    parser.add_argument('--max-moves', '-m', type=int, default=60, help='Max moves per game.')
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--board-out')
    parser.add_argument('--move-out')

    args = parser.parse_args()
    xs = []
    ys = []

    for i in range(args.num_games):
        print('Generating game %d/%d...' % (i + 1, args.num_games))
        x, y = generate_game(args.board_size, args.rounds, args.max_moves, args.temperature)
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    np.save(args.board_out, x)
    np.save(args.move_out, y)

if __name__ == '__main__':
    main()