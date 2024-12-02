from copy import deepcopy
import random
import time
import itertools

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

        # Choose the most visited child as the best move
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child

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
        """
        Implements a default policy to select a simulation move.
        Checks for basic savebridge template connections to play deterministically.
        """
        savebridge_moves = self.savebridge(board, colour, legal_moves)
        if len(savebridge_moves) > 0:
            return random.choice(savebridge_moves)

        return random.choice(legal_moves)
    
    def savebridge(self, board: Board, colour: Colour, legal_moves: list[Move]) -> Move:
        """
        Implements the savebridge function to identify and play moves
        that secure connections using the bridge and wheel templates.
        Based on the template connection strategy defined in:
        https://webdocs.cs.ualberta.ca/~mmueller/ps/2013/2013-CG-Mohex2.0.pdf

        Args:
            board (Board): The current state of the board.
            colour (Colour): The player's colour.
            legal_moves (list[Move]): A list of legal moves.

        Returns:
            Move | None: The move to secure a template connection, or None if no template applies.
        """
        # Check for bridge opportunities
        savebridge_moves = []
        for move in legal_moves:
            x, y = move.x, move.y
            
            # Look for the bridge template
            neighbors = Utilities.get_neighbours(board, x, y)
            for n1, n2 in itertools.combinations(neighbors, 2):
                # Check if n1 and n2 form a bridge with the current move
                if (n1.colour == colour and
                    n2.colour == colour and
                    self.is_bridge(n1, n2, move)):
                    savebridge_moves.append(move)
                
        # Check for wheel opportunities
        for move in legal_moves:
            x, y = move.x, move.y
            
            # Check if the move completes a wheel template
            if self.completes_wheel(board, x, y, colour):
                savebridge_moves.append(move)
        
        return savebridge_moves
    
    def is_bridge(self, n1, n2, move):
        """
        Checks if n1 and n2, with the given move, form a bridge.
        
        Args:
            n1 (Tile): The first neighbor.
            n2 (Tile): The second neighbor.
            move (Move): The move being evaluated.
        
        Returns:
            bool: True if the pattern forms a bridge, False otherwise.
        """
        # Bridge condition: Diagonal adjacency with one gap
        return (abs(n1.x - n2.x) == 2 and abs(n1.y - n2.y) == 2 and
                (move.x, move.y) == ((n1.x + n2.x) // 2, (n1.y + n2.y) // 2))
    
    def completes_wheel(self, board: Board, x: int, y: int, colour: Colour) -> bool:
        """
        Checks if placing a tile at (x, y) completes a wheel template.
        
        Args:
            board (Board): The game board.
            x (int): X-coordinate of the move.
            y (int): Y-coordinate of the move.
            colour (Colour): The player's colour.
        
        Returns:
            bool: True if the move completes a wheel, False otherwise.
        """
        neighbors = Utilities.get_neighbours(board, x, y)
        same_coloured_neighbors = [
            neighbour for neighbour in neighbors
            if neighbour.colour == colour
        ]
        
        # Wheel condition: At least 3 neighbors of the same colour
        return len(same_coloured_neighbors) >= 3




######################################################################################