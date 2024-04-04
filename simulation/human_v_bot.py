from go import agent
from go import goboard_slow as goboard
from go import gotypes
from go.utils import print_board, print_move, point_from_coords
import time

def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bots = agent.RandomBot()
    while not game.is_over():
        time.sleep(0.3)
        print(chr(27)  +  "[2J")
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bots.select_move(game)
        print_move(game.next_player,move)
        game = game.apply_move(move)

if __name__ == '__main__':
    main()