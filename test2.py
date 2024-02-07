from go.move_generation import Go, Stone, POINT_STATES

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
    game.play('9j')
    game.play('9h')
    game.play('8h')
    game.play('8j')
    game.print_board()
    
    check_group = input('coordinate: ')
    row, col = (game.get_row(int(check_group[0])),game.get_col(check_group[1].upper()))
    print(game.count_liberties(game.intersection[row][col],row,col))
    

def territory(board):
    answer = {'B': 0, 'W': 0}
    checked = []
    stack = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == '+' and (i, j) not in checked:
                stack.append((i, j))
                checked.append((i, j))
                count = 0
                color = set()
                while stack:
                    y, x = stack.pop(0)
                    count += 1
                    for p, q in (y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x):
                        if 0 <= p < len(board) and 0 <= q < len(board):
                            if (p, q) not in checked:
                                if board[p][q] == '+':
                                    stack.append((p, q))
                                    checked.append((p, q))
                                else:
                                    color.add(board[p][q])
                if len(color) == 1:
                    answer[color.pop()] += count
    return answer


if __name__ == '__main__':
    print("Example:")
    print(territory(['++BW+++++',
                     '+BBW+++++',
                     'BWWWWWWWW',
                     'WWWWWB+++',
                     '++WBBB+++',
                     '++WWW++WW',
                     '++WWWWW++',
                     'WWWWW++++',
                     '+W+++++++']))

