import numpy as np

from keras.optimizers import SGD
from go.agent.base import Agent
from go.goboard import GameState, Move

class Branch:
    def __init__(self, prior):
        """this object store value of each branch"""
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
    
class ZeroTreeNode:
    """MCTS node"""
    def __init__(self, state: GameState, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches: dict[Move, Branch] = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children: dict[Move, 'ZeroTreeNode'] = {}
    
    def moves(self):
        return self.branches.keys()
    
    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children
    
    def get_child(self, move):
        return self.children[move]
    
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count
    
    def prior(self, move):
        if move in self.branches:
            return self.branches[move].prior
        return None

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

class ZeroAgent(Agent):
    """MCTS Agent for AlphaZero"""
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = rounds_per_move
        self.c = c

    def select_move(self, game_state):
        pass
    
    def set_collector(self):
        pass

    def select_branch(self, node):
        pass

    def create_node(self, game_state: GameState, move=None, parent=None):
        pass

    def train(self, experience, learning_rate, batch_size):
        pass

