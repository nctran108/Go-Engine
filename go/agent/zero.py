import numpy as np

from keras.optimizers import SGD
from go.agent.base import Agent
from go.goboard import GameState, Move
from go import kerasutil
from go.utils import print_board

from go.encoders import get_encoder_by_name

class Branch:
    def __init__(self, prior):
        """this object store value of each branch"""
        self.prior = prior # probability of the move
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
    
    def get_move(self, point):
        moves = self.moves()
        for move in moves:
            if move.point == point:
                return move
            elif move == Move.pass_turn:
                return Move.pass_turn
        return None
    
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
        return self.branches[move].prior

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
        # num_moves controls the number of times you repeat the search process
        self.num_rounds = rounds_per_move 
        self.c = c

    def select_move(self, game_state: GameState):
        root = self.create_node(game_state)
        #print_board(game_state.board)

        for i in range(self.num_rounds): # this is the first step in a process that repeats many times per move
            node = root
            next_move = self.select_branch(node) # select move that have max score
            # move down to bottom of the branch
            while node.has_child(next_move) and (not node.state.is_over()): # when has child return False, you've reach the bottom of the tree
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
            
            # end walk down and start back up
            # this is back propagation steps to walk back to the top of the tree and update all nodes
            new_state = node.state.apply_move(next_move)
            move = next_move
            child_node = self.create_node(new_state, parent=node)
            
            value = -1 * child_node.value # each level in the tree, you switch perspective between the two players.
                                          # therefore, you must multiply the value by -1.
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value
        
        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([root.visit_count(self.encoder.moves[idx])\
                                      for idx in range(self.encoder.num_moves())])
            self.collector.record_decision(root_state_tensor, visit_counts)

        return max(root.moves(), key=root.visit_count)
    
    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node: ZeroTreeNode):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)
        return max(node.moves(), key=score_branch)

    def create_node(self, game_state: GameState, move=None, parent: ZeroTreeNode=None) -> ZeroTreeNode:
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)

        priors = priors[0]
        # add Dirichlet noise to the root node
        if parent is None:
            noise = np.random.dirichlet(0.03 * np.ones_like(priors))
            priors = 0.75 * priors + 0.25 * noise
        
        value = values[0][0]

        move_priors = {self.encoder.moves[idx]: p for idx,p in enumerate(priors)}

        new_node = ZeroTreeNode(game_state, value,
                                move_priors,
                                parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]

        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape(num_examples, 1)
        action_target = experience.visit_counts / visit_sums

        value_target = experience.rewards

        self.model.compile(SGD(learning_rate=learning_rate),
                           loss=['categorical_crossentropy', 'mse'])
        self.model.fit(model_input,
                       [action_target,value_target],
                       batch_size=batch_size)
        
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_size'] = self.encoder.board_size
        h5file.create_group('rounds_per_move')
        h5file['rounds_per_move'].create_dataset('value', data=self.num_rounds)
        h5file.create_group('c')
        h5file['c'].create_dataset('value', data=self.c)
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

def load_zero_agent(h5file):
    print(h5file['model'])
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_size = h5file['encoder'].attrs['board_size']
    encoder = get_encoder_by_name(encoder_name,
                                 board_size)
    rounds_per_move = np.array(h5file['rounds_per_move']['value'])
    c = np.array(h5file['c']['value'])
    return ZeroAgent(model, encoder, rounds_per_move, c)