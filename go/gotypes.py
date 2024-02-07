import enum
from collections import namedtuple

class Player(enum.Enum):
    """enum player color"""
    black = 1
    white = 2

    @property
    def other(self):
        """select other color"""
        return Player.black if self == Player.white else Player.white
    
class Point(namedtuple('Point', 'row col')):
    """this class is an immutable and tuple-like data structure that preresentation the intersect/point on the board game with row and col"""
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1)
        ]