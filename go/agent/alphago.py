import numpy as np
from go.agent.base import Agent
from go.goboard import Move
from go import kerasutil
import operator

__all__ = ['AlphaGoNode',
           'AlphaGoMCTS']

class AlphaGoNode:
    def __init__(self, parent: 'AlphaGoNode'=None, probability=1.0):
        self.parent = parent
        self.children = {}

        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability
        self.u_value = probability

    def select_child(self):
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + child[1].u_value)

    def expand_children(self, moves, probabilities):
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(probability=prob)

    def update_values(self, left_value):
        if self.parent is not None:
            self.parent.update_values(left_value)

        self.visit_count += 1

        self.q_value += left_value / self.visit_count

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) * self.prior_value / (1 + self.visit_count)


class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                 lambda_value=0.5, num_simulations=1000,
                 depth=50, rollout_limit=100):
        self.policy_agent = policy_agent
        self.rollout_policy_agent = fast_policy_agent
        self.value_agent = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()

    def select_move(self, game_state):
        for _ in range(self.num_simulations):
            current_state = game_state
            node = self.root
            for _ in range(self.depth):
                if not node.children:
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)
                    node.expand_children(moves, probabilities)

                move, node = node.select_child()
                current_state = current_state.apply_move(move)

            value = self.value_agent.predict(current_state)
            rollout = self.policy_rollout(current_state)

            weighted_value = self.lambda_value * rollout + (1 - self.lambda_value) * value

            node.update_values(weighted_value)

        # Choose the move with the highest visit count
        best_move = max(self.root.children,
                        key=lambda move: self.root.children.get(move).visit_count)

        self.root = AlphaGoNode()  # Reset the root for the next move
        if best_move in self.root.children:
            self.root = self.root.children[best_move]  # Move the root to the selected child
            self.root.parent = None  # Detach the new root from its parent

        return best_move

    def policy_probabilities(self, game_state):
        encoder = self.policy_agent.encoder
        outputs = self.policy_agent.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs

    def policy_rollout(self, game_state):
        for _ in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probs = self.rollout_policy_agent.predict(game_state)
            encoder = self.rollout_policy_agent.encoder
            valid_moves = [m for idx, m in enumerate(move_probs)
                           if Move(encoder.decode_point_index(idx)) in game_state.legal_moves()]
            max_index, max_value = max(enumerate(valid_moves), key=operator.itemgetter(1))
            max_point = encoder.decode_point_index(max_index)
            greedy_move = Move(max_point)
            if greedy_move in game_state.legal_moves():
                game_state = game_state.apply_move(greedy_move)

        next_player = game_state.next_player
        winner = game_state.winner()
        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0

    def serialize(self, h5file):
        raise IOError("AlphaGoMCTS agent can\'t be serialized" +
                       "consider serializing the three underlying" +
                       "neural networks instad.")