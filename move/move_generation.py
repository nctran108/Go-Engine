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
    
    def count_liberties(self, color):
        pass

    def rule_state_machine(self):
        pass

    def play(self,move):
        pass

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
    game.print_board()

if __name__ == "__main__":
    main()
