from move_generation import Go
from score import Score

def main():
    game = Go(9)
    game.print_board()
    
    while (not game.end):
        move = input('next move: ')
        if game.play(move):
            game.print_board()

    score = Score(game)
    

if __name__ == "__main__":
    main()