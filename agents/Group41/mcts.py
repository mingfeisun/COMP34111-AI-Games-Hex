import math
from random import choice
from time import time
from typing import List, Tuple

# from agents.Group41.board_state import Board, Move

class Node:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}      # Move -> Node
        self.visits = 0         # N
        self.total_value = 0    # W (total value)
        self.prior = prior      # P (probability from neural network)

    @property
    def value(self) -> int:
        """Returns Q (Expected win rate)"""
        return self.total_value / self.visits if self.visits != 0 else 0

    def is_leaf(self) -> bool:
        """A node with no children is a leaf node"""
        return len(self.children) == 0

    def expand(self, priors: List[(Move, int)]) -> None:
        """
        Expands node by creating children
        priors: List of (move, prob) tuples from neural network
        """
        for move, prob in priors:
            if move not in self.children:
                self.children[move] = Node(parent=self, prior=prob)

    def update(self, value: int) -> None:
        """Updates node stats during backpropagation"""
        self.visits += 1
        self.total_value += value

    def most_visits(self) -> Move:
        """Returns move with highest visit count"""
        return max(self.children.items(), key=lambda item: item[1].visits)[0]

class MCTS:
    def __init__(self, game):
        self.game = game
        # TODO: add param for NN model and maybe extra args 
        # e.g c_puct constant used by AlphaZero for Polynomial
        # Upper Confidence Trees (PUCT) algorithm

    def search(self, board: Board, time_limit: int) -> Node:
        """
        Main MCTS Loop

        Args:
            board (Board): The current board state
            time_limit (int): Time allowed for move in seconds
        """
        root = Node()

        start = time()
        end = time()

        while end - start < time_limit:

            # 1: Selection
            node = root
            search_board = board.copy()

            while not node.is_leaf():
                move, node = self.find_best_child(node)
                search_board.play_move(move)
            
            # 2: Expansion
            value, finished = search_board.get_result()

            if not finished:
                # if self.model:
                #   TODO: Neural network implementation
                #   priors, value = self.mode.predict(search_board)

                # FALLBACK: uniform priors (1 / num_moves)
                legal_moves = search_board.get_legal_moves()
                prob = 1.0 / len(legal_moves)
                priors = [(move, prob) for move in legal_moves]
                node.expand(priors)

            # 3: Simulation
            if not finished:
                # if self.model:
                #   TODO: Neural network implementation
                #   pass (value came from step 2 with NN)

                # FALLBACK: play random moves until end of game to get a result
                value = self.random_simulation(search_board)
            
            # 4: Backpropagation
            while node is not None:
                node.update(value)
                value = -value
                node = node.parent
            
        return root.most_visits()

    def find_best_child(self, node: Node) -> Tuple[Move, Node]:
        """Choose child with highest PUCT"""
        best_score = float('-inf')
        best_move = None
        best_child = None

        sqrt_parent_visits = math.sqrt(node.visits)

        for move, child in node.children.items():
            q = child.value
            u = self.c_puct * child.prior * (sqrt_parent_visits / (1 + child.visits))
            score = q + u

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def random_simulation(self, board: Board) -> int:
        """Random rollout fallback for simulation phase"""
        # TODO
        return 0