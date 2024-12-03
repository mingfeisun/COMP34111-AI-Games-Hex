from copy import deepcopy
import random
import time

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group23.utilities import Utilities
from agents.Group23.treenode import TreeNode

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, board: Board, colour: Colour, turn_length_s: int = 5):
        self.board = board  # The game board
        self.colour = colour  # Agent's colour
        self.turn_length = turn_length_s  # Length of a MCTS search in seconds

    def run(self, root: TreeNode):
        """Performs MCTS simulations from the root node."""
        iterations = 0
        start_time = time.time()
        while time.time() - start_time < self.turn_length:
            iterations += 1
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)

        print(f'Ran {iterations} simulations in {time.time() - start_time:.2f}s')

        # Choose the most visited child as the best move
        best_child = max(root.children, key=lambda child: child.wins / child.visits)
        print(f'Selected move with {best_child.visits} visits and {best_child.wins} wins')
        return best_child

    def _select(self, node: TreeNode):
        """Selects a node to expand using the UCT formula."""
        moves = self.get_heuristic_moves(self.board)
        while node.is_fully_expanded(moves):
            node = node.best_child()
        return self._expand(node)

    def _expand(self, node: TreeNode):
        """Expands the node by adding a new child."""
        legal_moves = self.get_heuristic_moves(self.board)
        unvisited_moves = [move for move in legal_moves if move not in [child.move for child in node.children]]

        if len(unvisited_moves) > 0:
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
            moves = self.get_heuristic_moves(simulation_board)

            move = self._default_policy(moves)

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

    def get_all_moves(self, board: Board) -> list[Move]:
        available_tiles = []
        for row in board.tiles:
            for tile in row:
                if tile.colour != Colour.RED and tile.colour != Colour.BLUE:
                    available_tiles.append(tile)
        
        if len(available_tiles) == 0:
            raise ValueError("No legal moves available")
        
        return [(tile.x, tile.y) for tile in available_tiles]
    
    def get_heuristic_moves(self, board: Board) -> list[Move]:
        """
        Generates a subset of all legal moves for the current board state, based on the heuristic given:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4406406
        """
        moves = Utilities.find_one_to_connect_moves(board, self.colour)

        if len(moves) == 0:
            moves = self.get_all_moves(board)

        return [Move(x, y) for x, y in moves]

    def _default_policy(self, moves: list[Move]) -> Move:
        """
        Implements a default policy to select a simulation move.
        """
        return random.choice(moves)