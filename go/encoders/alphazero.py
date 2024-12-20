from go.encoders import Encoder
from go.goboard import Move, Point, Player, GameState

import numpy as np

class ZeroEncoder(Encoder):
    def __init__(self, board_size):
        self.board_size = board_size
         # 0 - 3. our stones with 1, 2, 3, 4+ liberties
        # 4 - 7. opponent stones with 1, 2, 3, 4+ liberties
        # 8. 1 if we get komi
        # 9. 1 if opponent gets komi
        # 10. move would be illegal due to ko
        self.num_planes = 11
        self.moves = []
        for idx in range(self.num_moves()):
            self.moves.append(self.decode_move_index(idx))

    def name(self):
        return 'alphazero'

    def encode_move(self, move: Move):
        if move.is_play:
            return (self.board_size * (move.point.row - 1) + (move.point.col -1))
        elif move.is_pass:
            return self.board_size * self.board_size
        raise ValueError('Cannot encode resign move')
    
    def decode_move_index(self, index):
        if index == self.board_size * self.board_size:
            return Move.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Move.play(Point(row=row+1,col=col+1))
    
    def decode_point_index(self, index):
        if index == self.board_size * self.board_size:
            return None
        row = index // self.board_size
        col = index % self.board_size
        return Point(row=row+1,col=col+1)

    def num_moves(self):
        return self.board_size * self.board_size + 1
    
    def encode(self, game_state: GameState):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
        if game_state.next_player == Player.white:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(next_player,
                                                       Move.play(p)):
                        board_tensor[10][r][c] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color != next_player:
                        liberty_plane += 4
                    board_tensor[liberty_plane][r][c] = 1
        return board_tensor
    
    def shape(self):
        return self.num_planes, self.board_size, self.board_size
    
def create(board_size):
    return ZeroEncoder(board_size)