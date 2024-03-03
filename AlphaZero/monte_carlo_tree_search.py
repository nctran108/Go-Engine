from go.gotypes import Player
from go.agent import Agent, naive
import random
import math

class MCTSNode(object):
    def __init__(self, game_state, parent: 'MCTSNode'=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0
        }
        self.num_rollouts = 0
        self.children: list[MCTSNode] = []
        self.unvisited_moves = game_state.legal_move()

    def add_random_child(self):
        index = random.randint(0,len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node
    
    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        """reports whether this position has any legal moves
          that haven't yet been added to the tree"""
        return len(self.unvisited_moves) > 0
    
    def is_terminal(self):
        """reports whether the game is over at this node;
        if so,, you can't research any further from here"""
        return self.game_state.is_over()
    
    def winning_frac(self, player):
        """return the fraction of rollouts that were won by a given player"""
        return float(self.win_counts[player]) / float(self.num_rollouts)
    
    def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
        """this fusnsction calculate the uct score. utc = w + c*sqrt(log(N)/n)"""
        exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
        return win_pct + temperature * exploration

class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)
            
            # add a new child node into the tree.
            if node.can_add_child():
                node = node.add_random_child()
            
            # simulate a random game for this node
            winner = self.simulate_random_game(node.game_state)

            # propagate scores back up the tree            
            while node is not None:
                node.record_win(winner)
                node = node.parent
        # pick a move
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_percentage = child.winning_frac(game_state.next_player)
            if child_percentage > best_pct:
                best_pct = child_percentage
                best_move = child.move
                print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move
    
    def select_child(self, node: MCTSNode):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        best_score = -1
        best_child = None
        for child in node.children:
            score = MCTSNode.uct_score(total_rollouts,child.num_rollouts, 
                                       child.winning_frac(node.game_state.next_player),
                                       self.temperature)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.black: naive.RandomBot(),
            Player.white: naive.RandomBot()
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()

