import sys
import os
sys.path.append(os.getcwd())

from go import agent
from go import goboard_slow
from go import gotypes
from go.utils import print_board, print_move
import time

def main():
    board_size = 9
    game = goboard_slow.GameState.new_game(board_size)
    bot = agent.load_zero_agent("bots/zero_demo_9x9_5_games_1600.weights.h5", 'r')
    bots = {
        gotypes.Player.black: bot,
        gotypes.Player.white: bot
    }
    while not game.is_over():
        time.sleep(0.3)
        print(chr(27)  +  "[2J")
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player,bot_move)
        game = game.apply_move(bot_move)

if __name__ == '__main__':
    main()