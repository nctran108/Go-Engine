import unittest

from go.agent.helpers import is_point_an_eye
from go.goboard import Board, GameState, Move
from go.gotypes import Player, Point
from go.encoders.alphago import AlphaGoEncoder
from go.utils import print_board


class AlphaGoEncoderTest(unittest.TestCase):
    def test_encoder(self):
        alphago = AlphaGoEncoder((19,19),True)

        start = GameState.new_game(19)
        next_state = start.apply_move(Move.play(Point(16, 16)))
        next_state = next_state.apply_move(Move.play(Point(15, 16)))
        next_state = next_state.apply_move(Move.play(Point(15, 15)))
        next_state = next_state.apply_move(Move.play(Point(16, 15)))
        next_state = next_state.apply_move(Move.play(Point(16, 17)))
        next_state = next_state.apply_move(Move.play(Point(17, 15)))
        next_state = next_state.apply_move(Move.play(Point(14, 16)))
        next_state = next_state.apply_move(Move.play(Point(15, 17)))
        next_state = next_state.apply_move(Move.play(Point(15, 18)))
        #next_state = next_state.apply_move(Move.play(Point(17, 16)))
        next_state = next_state.apply_move(Move.play(Point(14, 17)))
        feature = alphago.encode(next_state)

        self.assertEqual(alphago.name(), 'alphago')
        self.assertEqual(alphago.board_height, 19)
        self.assertEqual(alphago.board_width, 19)
        self.assertEqual(alphago.num_planes, 49)
        self.assertEqual(alphago.shape(), (49, 19, 19))


        try:
            print_board(next_state.board)
            print()
            print(feature[46])
        except (Exception,TypeError) as e:
            print(e)
        label = alphago.encode_point(Point(16,16))


if __name__ == '__main__':
    unittest.main()