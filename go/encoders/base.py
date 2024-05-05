import importlib

class Encoder:
    def name(self):
        """support logging or saving the name of the encoder model is using"""
        raise NotImplementedError()
    
    def encode(self, game_state):
        """turns a Go board into numeric data"""
        raise NotImplementedError()
    
    def encode_point(self, point):
        """Turns a Go point into an integer index"""
        raise NotImplementedError()
    
    def decode_point_index(self, index):
        """Turns an integer index back into a Go point"""
        raise NotImplementedError()
    
    def num_points(self):
        """Number of points on the board--board width times board height"""
        raise NotImplementedError()
    
    def shape(self):
        """Shape of the encoded board structure"""
        raise NotImplementedError()
    
def get_encoder_by_name(name, board_size):
    if isinstance(board_size, int):
        if name != 'alphazero':
            board_size = (board_size, board_size)
    module = importlib.import_module('go.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)