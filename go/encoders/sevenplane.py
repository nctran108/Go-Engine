import numpy as np

from go.encoders.base import Encoder
from go.goboard import Move, Point
from go.gotypes import Player

class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size):
        """
        this encoder contain 7 planes:
        - the first plane has a 1 for every white stone that has precisely one liberty, and 0s otherwise
        - the second and third feature planes have a 1 for white stones with 2 or at least 3 liberties.
        - the 4th to 6th planes do the same for black stones.
        - the last feature plane marks points that can't be played because of ko with a 1.
        """
        self.board_width, self.board_height = board_size
        self.num_planes = 7

    def name(self):
        return 'sevenplane'
    
    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        base_plane = {Player.white: 0,
                      Player.black: 3}
        for row in range(self.board_height):
            for col in range(self.board_width):
                p = Point(row=row + 1, col=col + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player, Move.play(p)):
                        board_tensor[6][row][col] = 1
                    else:
                        liberty_plane = min(3, go_string.num_liberties) - 1
                        liberty_plane += base_plane[go_string.color]
                        board_tensor[liberty_plane][row][col] = 1
        return board_tensor
    
    def encode_point(self, point: Point):
        return self.board_width * (point.row - 1) + (point.col - 1)
    
    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row+1,col=col+1)
    
    def num_points(self):
        return self.board_width * self.board_height
    
    def shape(self):
        return self.num_planes, self.board_height, self.board_width
    
def create(board_size):
    return SevenPlaneEncoder(board_size)