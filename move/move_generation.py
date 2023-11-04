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
        self.intersection = []
        self.black_turn = True
        self.end = False
        self.black_captured = 0
        self.white_captured = 0
        self.both_passed = False

        self.board, self.horizontal, self.vertical = self.create_board()
        self.previous_board = None

    def create_board(self) -> Tuple[List,List,List]:
        """
        this function create the Go board base on the size of the board
        Args: None
        Return:
            turple of list of the board, verticle, and horizontal
        """
        board = []
        horizontal = []
        vertical = []
        original = 'A'
        # go through loop and create the 2D board
        for i in range(self.board_size):
            stone = []
            point = []
            for j in range(self.board_size):
                stone.append(Stone().EMPTY)
                point.append(POINT_STATES().EMPTY)
            board.append(stone)
            self.intersection.append(point)
            # create horizontal label
            horizontal.append(self.board_size - i)
            letter = ord(original) + i

            # generate label for vertical
            if letter == ord('I'):
                vertical.append('J')
            elif letter < ord('I'):
                vertical.append(chr(letter))
            else:
                vertical.append(chr(letter + 1))
        return (board, horizontal, vertical)

    def get_row(self, value):
        """ this function return the row of which horizontal line"""
        return self.horizontal.index(value)

    def get_col(self, value):
        """ this function return the col of the selected vertical line"""
        return self.vertical.index(value)
    
    def get_board(self):
        """ this function return the board"""
        return self.board, self.horizontal, self.vertical
    
    def count_liberties(self, ocupied: POINT_STATES, row, col):
        """this function count the liberties of the occupied stone group
        ARGS:
            POINT_STATES: the board intersection
            int: ROW of the board
            int: col of the board
        return:
            int: the number of liberties
        """
        if ocupied == Stone().EMPTY:
            return 0
        # get the whole group
        group = self.get_group(ocupied, row, col)
        # check if the stone is visited or not
        visited = []

        def flood_fill(r,c):
            """ this is recursive function to count the liberties of each stone in every direction"""
            if not self.is_valid_move(r, c):
                return 0

            # skip if visited
            if (r,c) in visited:
                return 0
            # else just mark as visited
            visited.append((r,c))
            
            # check if the intersection is empty then count 1
            if (self.intersection[r][c] == POINT_STATES().EMPTY):
                return 1
            # check if the intersection is occupied but not in same group then not count
            if (self.intersection[r][c] != POINT_STATES().EMPTY) and (r,c) not in group:
                return 0

            # if the stone in the group and not empty then count the liberties of that stone in 4 directions
            liberties = 0
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                new_row = r + dr
                new_col = c + dc
                # count the liberties and recursive
                liberties += flood_fill(new_row,new_col) 
            return liberties

        # count liberties of the stone
        liberties = flood_fill(row,col)
        
        return liberties
    
    def is_valid_move(self,row, col):
        """this function check is the row and col is in bound"""    
        return (0 <= row and row < self.board_size) and (0 <= col and col < self.board_size)
    
    def get_group(self, ocupied: POINT_STATES, row, col):
        """this function get the whole group of the selected stone"""
        group = [] # empty group
        
        # not the correct color then this is not right group
        if (self.intersection[row][col] != ocupied):
            return []
        
        def count_group(r,c,group):
            """recursive function to get the group"""
            # check the row and col bound or the stone is visited
            if not self.is_valid_move(r,c) or self.intersection[r][c] != ocupied or (r,c) in group:
                return
            # if valid and not visited then add in  to the group
            group.append((r,c))
            # check 4 direction of the stone
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                new_row = r + dr
                new_col = c + dc
                count_group(new_row,new_col,group) 
        # call recursive function to get the group
        count_group(row,col,group)
        return group
    
    def capture_stones(self,row, col):
        direction = [(-1,0),(1,0),(0,-1),(0,1)]
        # get opposite player
        opposite_color = self.opposite_player()

        # check on every direction to capture
        for dr,dc in direction:
            new_row = row + dr
            new_col = col + dc
            # if the move is valid and the opponent liberties is zero then capture
            if self.is_valid_move(new_row,new_col) and self.board[new_row][new_col] == opposite_color:
                if self.count_liberties(self.intersection[new_row][new_col], new_row, new_col) == 0:
                    self.remove_group(new_row,new_col)

    def remove_group(self, row, col):
        # check is the stone is empty
        if self.board[row][col] == Stone().EMPTY:
            return
        
        # get the whole group
        groups = self.get_group(self.intersection[row][col],row,col)

        # remove whole group
        for stone_row, stone_col in groups:
            self.board[stone_row][stone_col] = Stone().EMPTY
            self.intersection[stone_row][stone_col] = POINT_STATES().EMPTY

    
    def current_player(self):
        """this function reture the current player as stone"""
        if self.black_turn:
            return Stone().BLACK
        else:
            return Stone().WHITE
    
    def opposite_player(self):
        """this function return the opponent player as stone"""
        if self.black_turn:
            return Stone().WHITE
        else:
            return Stone().BLACK
        
    def occupied(self, stone):
        """this function return the correct intersection of the board"""
        if stone == Stone().BLACK:
            return POINT_STATES().OCCUPIED_BY_BLACK
        elif stone == Stone().WHITE:
            return POINT_STATES().OCCUPIED_BY_WHITE
        else:
            return POINT_STATES().EMPTY

    def make_move(self, row, col):
        if self.is_valid_move(row,col):
            # place stone on the board
            self.board[row][col] = self.current_player()

            # mark the intersection accupied 
            self.intersection[row][col] = self.occupied(self.board[row][col])

            # check the stone liberties
            liberties = self.count_liberties(self.intersection[row][col],row,col)

            if liberties == 0:
                # check for the ko rule
                if self.is_ko():
                    self.board[row][col] = Stone().EMPTY
                    self.intersection[row][col] = self.occupied(self.board[row][col])
                    self.previous_board = None
                    return False
                # create copy board
                self.previous_board = [row[:] for row in self.board]
                
            # check for capture and update the board
            self.capture_stones(row,col)

            # Switch the current player
            self.black_turn = not self.black_turn
            return True
        else:
            return False
        
    def is_ko(self):
        """this function check the ko rule to see if the move is valid"""
        if self.previous_board is None:
            return False
        return self.previous_board == self.board
    
    def player_passed(self,move):
        return move.upper() == "PASS"
    
    def score(self):
        return True

    def play(self,move : str):
        """this function use to play the game
        ARGS:
            str: player move
        Return:
            bool: the move is valid or not
        """
        # check the input move
        move = move.upper()
        if not self.move_allowed(move):
            return False

        # convert move label to row and col
        row = self.get_row(int(move[0]))
        col = self.get_col(move[1])

        # check if both player pass
        if self.player_passed(move):
            # check if both player passed

            if self.both_passed:
                # check score
                self.score()
                self.end = True
            else:
                # only one passed
                self.black_turn = not self.black_turn
            return True
        
        # if the intersection of that row and col is available then play
        if (self.intersection[row][col] == POINT_STATES().EMPTY):
            # make the move
            status = self.make_move(row,col)
            # return if the move is invalid
            if not status:
                print("[WARNING]: Invalid move")
                return False
        # else the intersect is occupied and can't play
        else:
            print("[WARNING]: the intersection is already occupied by", end=' ')
            if self.intersection[row][col] == POINT_STATES().OCCUPIED_BY_BLACK:
                print("back")
            else:
                print("white")
            return False
        
        return True
        

    def move_allowed(self,move : str):
        """this function check the player move to see if it is valid move"""
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
        """this function print out the whole board to display players stones"""
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
        """this function empty the whole board"""
        for row_index in range(self.board_size):
            for col_index in range(self.board_size):
                self.board[row_index][col_index] = Stone().EMPTY

    def print_intersect(self):
        """this function print out the whole board to display players stones"""
        index = 0
        for r in self.intersection:
            print(str(self.horizontal[index]).rjust(2, ' '), end=' ')
            for c in r:
                print(c,end=' ')
            print()
            index = index + 1
        print('  ',end=' ')
        for letter in self.vertical:
            print(letter,end=" ")
        print('\n')

def main():
    game = Go(9)
    game.print_board()
    
    while (not game.end):
        move = input('next move: ')
        if game.play(move):
            game.print_board()
        if move == 'end':
            game.end = True
    

if __name__ == "__main__":
    main()
