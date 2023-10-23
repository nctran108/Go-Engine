import sys
from typing import List, Tuple

class Stone:
    def __init__(self) -> None:
        self.WHITE = "o"
        self.BLACK = "#"
        self.LIBERTY = "x"
        self.EMPTY = '.'

class POINT_STATES:
    def __init__(self) -> None:
        self.EMPTY = 0
        self.OCCUPIED_BY_WHITE = 1
        self.OCCUPIED_BY_BLACK = 2
        

class Go:
    def __init__(self, size):
        self.board_size = size
        self.stones = Stone()
        self.position = POINT_STATES()
        self.liberties = []
        self.connected = []
        self.intersection = []
        self.black_turn = True
        self.end = False
        self.black_captured = 0
        self.white_captured = 0

        self.board, self.horizontal, self.vertical = self.create_board()

    def create_board(self) -> Tuple[List,List,List]:
        board = []
        horizontal = []
        vertical = []
        original = 'A'
        for i in range(self.board_size):
            stone = []
            point = []
            for j in range(self.board_size):
                stone.append(self.stones.EMPTY)
                point.append(self.position.EMPTY)
            board.append(stone)
            self.intersection.append(point)
            horizontal.append(self.board_size - i)
            letter = ord(original) + i
            if letter == ord('I'):
                vertical.append('J')
            elif letter < ord('I'):
                vertical.append(chr(letter))
            else:
                vertical.append(chr(letter + 1))
        return (board, horizontal, vertical)

    def get_row(self, value):
        return self.horizontal.index(value)

    def get_col(self, value):
        return self.vertical.index(value)
    
    def get_board(self):
        return self.board, self.horizontal, self.vertical
    
    def count_liberties(self, stone: Stone, coordinate: tuple):
        group = self.count_group(stone, coordinate)
        row, col = coordinate
        liberties = 0
        if not self.is_valid_move(row,col):
            return 0
        return liberties
    
    def get_group(self, stone: Stone, coordinate: tuple):
        
        def is_valid_move(row, col):
            return (0 <= row and row < self.board_size) and (0 <= col and col < self.board_size)
        
        
        row, col = coordinate
        group = []
        
        if (self.board[row][col] != stone):
            return []
        
        def count_group(r,c,group):
            if not is_valid_move(r,c) or self.board[r][c] != stone or (r,c) in group:
                return
            group.append((r,c))
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                new_row = r + dr
                new_col = c + dc
                count_group(new_row,new_col,group) 
                
        count_group(row,col,group)
        return group

    def play(self,move : str):
        move = move.upper()
        if not self.move_allowed(move):
            return False
        
        row = self.get_row(int(move[0]))
        col = self.get_col(move[1])
        
        if (self.intersection[row][col] == self.position.EMPTY):
            if self.black_turn:
                self.board[row][col] = self.stones.BLACK
                self.intersection[row][col] = self.position.OCCUPIED_BY_BLACK
                self.black_turn = False
            else:
                self.board[row][col] = self.stones.WHITE
                self.intersection[row][col] = self.position.OCCUPIED_BY_WHITE
                self.black_turn = True
        else:
            print("[WARNING]: the intersection is already occupied by", end=' ')
            if self.intersection[row][col] == self.position.OCCUPIED_BY_BLACK:
                print("back")
            else:
                print("white")
            return False
        
        return True
        

    def move_allowed(self,move : str):
        if len(move) == 2:
            if move[0].isdigit() and move[1].isalpha():
                if (int(move[0]) in self.horizontal) and (move[1] in self.vertical):
                    return True
                else:
                    print("move is not in the list")
                    return False
            print('wrong move')
            return False
        else:
            print('wrong move size')
            return False
        
    def print_board(self):
        index = 0
        for r in self.board:
            print(str(self.horizontal[index]).rjust(2, ' '), end=' ')
            for c in r:
                print(c,end=' ')
            print()
            index = index + 1
        print('  ',end=' ')
        for letter in self.vertical:
            print(letter,end=" ")
        print('\n')

    def clear_board(self):
        for row_index in range(self.board_size):
            for col_index in range(self.board_size):
                self.board[row_index][col_index] = self.stones.EMPTY


def main():
    game = Go(9)
    
    #while (not game.end):
    #    move = input('next move: ')
    #    if game.play(move):
    #        game.print_board()
    #    if move == 'end':
    #        game.end = True
    

if __name__ == "__main__":
    main()
