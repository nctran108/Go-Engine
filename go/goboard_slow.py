import copy
from go.gotypes import Player, Point

class Move():
    """this class decide which type of move on the game"""
    def __init__(self, point=None, is_pass= False, is_resign=False) -> None:
        assert(point is not None) ^ is_pass ^ is_resign

        self.point = point
        self.is_play = self.point is not None
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point=point)
    
    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)
    
    @classmethod
    def resign(cls):
        return Move(is_resign=True)
    
class GoString():
    """this is a string type of go that store color, stones, and liberties of group of stones"""
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)
    
    def remove_liberty(self, point):
        """remove liberties of removed stoned"""
        self.liberties.remove(point)

    def add_liberty(self, point):
        """add liberty from added stone"""
        self.liberties.add(point)

    def merged_with(self, go_string: object):
        """merge two group together and recount new group liberties"""
        assert go_string.color == self.color

        combined_stones = self.stones | go_string.stones

        return GoString(
                        self.color,
                        combined_stones,
                        (self.liberties | go_string.liberties) - combined_stones # A U B - C
        )
    
    @property
    def num_liberties(self):
        """return the number of liberties"""
        return len(self.liberties)
    
    def __eq__(self, other: object) -> bool:
        """for using '=' to compare both group"""
        return isinstance(other, GoString) and \
                self.color == other.color and \
                self.stones == other.stones and \
                self.liberties == other.liberties

class Board():
    """this class decide how to place each stone on the board"""
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {} # this store each point on the board

    def place_stone(self, player, point : Point):
        """this function how to place each stone on the board"""
        assert self.is_on_grid(point) # check if the stone place out side the grid
        assert self._grid.get(point) is None # check if the point is valid
        adjacent_same_color = [] # same color stone that visited
        adjacent_opposite_color = [] # other color stone that visited
        liberties = [] # liberties of placed stone
        # check neighbor of the stone
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue # if neighber is notside the board then skip
            # get the neighbor string if it exist
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None: # if neighber string is empty
                liberties.append(neighbor) # add the neighbor as the stone liberty
            elif neighbor_string.color == player: # if the neighbor is same color
                if neighbor_string not in adjacent_same_color: # if not visited yet
                    adjacent_same_color.append(neighbor_string) # add the neighbor to the asjacent same color
            else: # if different color
                if neighbor_string not in adjacent_opposite_color: # if not visited yet
                    adjacent_opposite_color.append(neighbor_string) # add the neighbor to adjacent opposite color
        
        # create new string that contain the player color, the current placed point, and its liberties
        new_string = GoString(player, [point], liberties)

        # for every visited same color, merge all same color into one group. this also call connect in Go
        for same_color_string in adjacent_same_color:
            new_string =  new_string.merged_with(same_color_string)
        # update each point on the grid with that new string
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string
        # for other color visited, remove the liberty of the other color stone
        for other_color_string in adjacent_opposite_color:
            other_color_string.remove_liberty(point)
        # for other color that visited, if that other color stone have no more liberties, just remove that stone out of the board
        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)
    
    def _remove_string(self, string: GoString):
        """this function remove strong from the board"""
        # for each point/stone in the string
        for point in string.stones:
            # check nenighbor of each stone
            for neighbor in point.neighbors():
                # get each neighber string
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None: # out side of the board
                    continue
                if neighbor_string is not string: # add back liberties to any group that neighber to this string (which is not this string)
                    neighbor_string.add_liberty(point)
            del(self._grid[point]) # remove the string

    def is_on_grid(self, point: Point):
        """this function check out of boundaries"""
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols
    
    def get(self, point: Point):
        """this function get the stone/group color"""
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color
    
    def get_go_string(self, point : Point):
        """this function get the whole group/ string"""
        string = self._grid.get(point)
        if string is None:
            return None
        return string
    
class GameState():
    """this class set the rulew for go game, this is link list"""
    def __init__(self, board : Board, next_player: Player, previous: 'GameState', move: Move):
        self.board = board # current board
        self.next_player = next_player # next color
        self.previous_state = previous  # the last game state
        self.last_move = move   # the last move

    def apply_move(self, move : Move):
        """this function apply move"""
        if move.is_play: # if still play, place the stone on the board
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else: # if pass or resign
            next_board = self.board # the board unchange
        return GameState(next_board,self.next_player.other, self, move)
    
    @classmethod
    def new_game(cls, board_size):
        """use this to empty the board and play new game."""
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
            board = Board(*board_size)
        return GameState(board, Player.black, None, None)
    
    def is_over(self):
        """this function check if the game is over"""
        if self.last_move is None: # check if the game begin
            return False
        if self.last_move.is_resign: # check if the player resign
            return True
        # check if both player passed
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass
    
    def is_move_self_capture(self, player: Player, move : Move):
        """this is to prevent self capture rule"""
        if not move.is_play: # the game ended
            return False
        # deep copy the board to test the next move
        next_board = copy.deepcopy(self.board)
        # try to place the stone on the temp board
        next_board.place_stone(player, move.point)
        # get the strong of that new move
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0 # return True of liberties equal 0
    
    @property
    def situation(self):
        return (self.next_player, self.board)
    
    def does_move_violate_ko(self, player : Player, move : Move):
        """this function check the ko rule"""
        if not move.is_play:
            return False
        # deep copy the board 
        next_board = copy.deepcopy(self.board)
        # play the stone on copy board to check the ko
        next_board.place_stone(player, move.point)
        # get the situation after place the stone
        next_situation = (player.other, next_board)
        # get previous state
        past_state = self.previous_state
        while past_state is not None: # this loop back to every state to check for the ko
            if past_state.situation == next_situation:
                return True
            past_state = past_state.previous_state
        return False
    
    def is_valid_move(self, move: Move):
        """this function check if the move is valid"""
        if self.is_over(): # game end
            return False
        if move.is_pass or move.is_resign: # the move either passed or resign
            return True
        # the move is not None, not self capture, and not violate the ko rule
        return (self.board.get(move.point) is None and
                 not self.is_move_self_capture(self.next_player, move) and 
                 not self.does_move_violate_ko(self.next_player, move))
    
    def legal_move(self):
        """this function return every legal moves on the board"""
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row,col))
                if self.is_valid_move(move):
                    moves.append(move)
        
        # this two moves are always legal
        moves.append(Move.pass_turn())
        moves.append(Move.resign())

        return moves
    
    def winner(self):
        if not self.is_over():
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = self.compute_game_result(self)
        return game_result.winner