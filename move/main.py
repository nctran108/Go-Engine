from move_generation import Go
from score import Score
import random

def main():
    board_size = int(input("board size: "))
    game = Go(board_size)
    game.print_board()

    move = str()
    computer_turn = False
    play_as = input("PLay as: ")

    if play_as == "WHITE":
        computer_turn = True

    while (not game.end):
        if computer_turn:
            row = random.randint(0, board_size - 1)
            col = random.randint(0, board_size - 1)
            move = str(game.horizontal[row]) + game.vertical[col]
            print("computer move: ", move)
        else:
            move = input('next move: ')
            ## temptory here to end the game
            if move == "pass":
                game.end = True

        if game.play(move):
            game.print_board()
            computer_turn = not computer_turn

    score = Score(game)
    

if __name__ == "__main__":
    main()