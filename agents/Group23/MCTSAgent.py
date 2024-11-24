import random
import math
import time
from copy import deepcopy

from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

class TreeNode:
    """Represents a node in the MCTS tree."""

    def __init__(self, move=None, parent=None):
        self.move = move  # The move that led to this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node

    def is_fully_expanded(self, legal_moves):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_param=math.sqrt(2)):
        """Selects the best child using UCT."""
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits) + exploration_param * math.sqrt(math.log(self.visits) / child.visits)
        )

    def add_child(self, move):
        """Adds a child node for a move."""
        child_node = TreeNode(move, parent=self)
        self.children.append(child_node)
        return child_node
    
######################################################################################

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, board: Board, colour: Colour, turn_length_s: int = 5):
        self.board = board  # The game board
        self.colour = colour  # Agent's colour
        self.turn_length = turn_length_s  # Length of a MCTS search in seconds

    def run(self, root: TreeNode):
        """Performs MCTS simulations from the root node."""
        start_time = time.time()
        while time.time() - start_time < self.turn_length:
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # Choose the most visited child as the best move
        return max(root.children, key=lambda child: child.visits).move

    def _select(self, node: TreeNode):
        """Selects a node to expand using the UCT formula."""
        legal_moves = self._get_legal_moves(self.board)
        while node.is_fully_expanded(legal_moves):
            node = node.best_child()
        return self._expand(node)

    def _expand(self, node: TreeNode):
        """Expands the node by adding a new child."""
        legal_moves = self._get_legal_moves(self.board)
        unvisited_moves = [move for move in legal_moves if move not in [child.move for child in node.children]]

        if unvisited_moves:
            new_move = random.choice(unvisited_moves)
            return node.add_child(new_move)

        return node

    def _simulate(self, node: TreeNode):
        """Simulates a random game from the current node and returns the result."""
        simulation_board = deepcopy(self.board)  # Create a copy of the board
        x, y = node.move.x, node.move.y
        simulation_board.set_tile_colour(x, y, self.colour)  # Play the move

        # Play randomly until the game ends
        current_colour = self.colour.opposite()
        while (not simulation_board.has_ended(colour=current_colour) and
               not simulation_board.has_ended(colour=current_colour.opposite())):
            legal_moves = self._get_legal_moves(simulation_board)

            move = self._default_policy(simulation_board, current_colour, legal_moves)

            x, y = move.x, move.y
            simulation_board.set_tile_colour(x, y, current_colour)
            current_colour = current_colour.opposite()

        return 1 if simulation_board.get_winner() == self.colour else 0

    def _backpropagate(self, node: TreeNode, result: int):
        """Backpropagates the simulation result through the tree."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # Invert the result for the opponent's perspective

    def _get_legal_moves(self, board: Board) -> list[Move]:
        available_tiles = []
        for row in board.tiles:
            for tile in row:
                if tile.colour != Colour.RED and tile.colour != Colour.BLUE:
                    available_tiles.append(tile)
        
        if len(available_tiles) == 0:
            raise ValueError("No legal moves available")
        
        return [Move(tile.x, tile.y) for tile in available_tiles]

    def _default_policy(self, board: Board, colour: Colour, legal_moves: list[Move]) -> Move:
        """Implements a default policy to select a simulation move."""


        return random.choice(legal_moves)



######################################################################################

class MCTSAgent(AgentBase):
    """An agent that uses MCTS for Hex."""

    def __init__(self, colour: Colour, turn_length_s: int = 5):
        super().__init__(colour)
        self.turn_length = turn_length_s # max length of a turn in seconds
        self.tree = None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Selects a move using MCTS."""
        if self.tree is None:
            self.tree = TreeNode()

        if opp_move is not None:
            # update game tree
            self.tree = self.update_tree(self.tree, opp_move)

            x, y = opp_move.x, opp_move.y
            board.set_tile_colour(x, y, self.colour.opposite())

        mcts = MCTS(board, self.colour, turn_length_s=self.turn_length)
        best_move = mcts.run(self.tree)

        x, y = best_move.x, best_move.y
        board.set_tile_colour(x, y, self.colour)

        # update game tree
        self.tree = self.update_tree(self.tree, best_move)

        return best_move
    
    def update_tree(self, tree: TreeNode, move: Move):
        """Updates the tree with the opponent's move."""
        for child in tree.children:
            if child.move == move:
                return child
        
        # If the move is not in the tree, create a new node
        return TreeNode(parent=tree, move=move)

