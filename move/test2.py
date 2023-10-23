from move_generation import Go, Stone, POINT_STATES

def main():
    game = Go(9)
    game.play('3g')
    game.play('7c')
    game.play('4f')
    game.play('6c')
    game.play('5g')
    game.play('5c')
    game.play('4c')
    game.play('5f')
    game.play('5e')
    game.play('6f')
    game.play('5d')
    game.play('6d')
    game.play('9a')
    game.play('5a')
    game.play('8a')
    game.play('1e')
    game.play('5j')
    game.play('6j')
    game.play('9b')
    game.play('6h')
    game.print_board()
    
    check_group = input('coordinate: ')
    coordinate = (game.get_row(int(check_group[0])),game.get_col(check_group[1].upper()))
    print(game.get_group(game.board[coordinate[0]][coordinate[1]],coordinate))
    
if __name__ == "__main__":
    main()