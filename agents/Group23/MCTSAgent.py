import random
import math
import time
from copy import deepcopy
import logging
import itertools

from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile

class Utilities:
    RELATIVE_NEIGHBOURS = [
        (-1, -1), (-1, 0), # row above
        (0, -1), (0, 1), # same row
        (1, 0), (1, 1) # row below
    ]

    def get_neighbours(board: Board, x, y) -> list[Tile]:
        """Returns a list of all neighbouring tiles."""
        neighbours = []
        for offset in Utilities.RELATIVE_NEIGHBOURS:
            x_offset, y_offset = offset
            x_n, y_n = x + x_offset, y + y_offset

            if Utilities.is_within_bounds(board, x_n, y_n):
                neighbours.append(board.tiles[x_n][y_n])
        
        return neighbours
    
    def is_within_bounds(board: Board, x, y) -> bool:
        """Checks if the coordinates are within the board bounds."""
        return 0 <= x < board.size and 0 <= y < board.size


######################################################################################

class TreeNode:
    """Represents a node in the MCTS tree."""

    def __init__(self, move=None, parent=None, player=None):
        self.move = move  # The move that led to this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node
        self.q_rave = 0  # times this move has been critical in a rollout
        self.n_rave = 0  # times this move has been played in a rollout
        self.player = player
        self.rave_const = 300
        
        if parent is not None:
            self.player = parent.player.opposite()

    def is_fully_expanded(self, legal_moves):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(legal_moves)

    def best_child(self, explore=math.sqrt(2)):
        """Selects the best child using UCT."""
        return max(
            self.children,
            key=lambda child: self.__value__(child, explore)
        )

    def add_child(self, move):
        """Adds a child node for a move."""
        child_node = TreeNode(move, parent=self)
        self.children.append(child_node)
        return child_node
    
    def __value__(self, child, explore=math.sqrt(2), heuristic='uct'):
        uct = (child.wins / child.visits) + explore * math.sqrt(math.log(self.visits) / child.visits)

        if heuristic == 'uct':
            return uct
        elif heuristic == 'rave':
            alpha = max(0, (self.rave_const - self.N) / self.rave_const)
            amaf = self.q_rave / self.n_rave if self.n_rave != 0 else 0
            rave = (1 - alpha) * uct + alpha * amaf
            return rave

        return self.wins / self.visits if self.visits != 0 else 0
    
######################################################################################

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
            result, self_rave_pts, opp_rave_points = self._simulate(node)
            self._backpropagate(node, result, self_rave_pts, opp_rave_points)

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
        """
        Simulates a random game from the current node and returns the result.
        Args:
            node (TreeNode): The node to simulate from.
        Returns:
            int: The result of the simulation. (-1 for loss, 1 for win)
        """
        simulation_board = deepcopy(self.board)  # Create a copy of the board
        x, y = node.move.x, node.move.y
        simulation_board.set_tile_colour(x, y, self.colour)  # Play the move

        # Play randomly until the game ends
        current_colour = self.colour.opposite()
        legal_moves = self._get_legal_moves(simulation_board)
        while (not simulation_board.has_ended(colour=current_colour) and
               not simulation_board.has_ended(colour=current_colour.opposite())):

            move = self._default_policy(simulation_board, current_colour, legal_moves)

            x, y = move.x, move.y
            simulation_board.set_tile_colour(x, y, current_colour)
            current_colour = current_colour.opposite()
            legal_moves.remove(move)

        opp_rave_pts = []
        self_rave_pts = []

        # Record critical cells for RAVE
        for x in range(simulation_board.size):
            for y in range(simulation_board.size):
                tile = simulation_board.tiles[x][y]
                if tile.colour == self.colour:
                    self_rave_pts.append((x, y))
                elif tile.colour == self.colour.opposite():
                    opp_rave_pts.append((x, y))
        
        game_value = 1 if simulation_board.get_winner() == self.colour else -1

        return game_value, self_rave_pts, opp_rave_pts

    def _backpropagate(self, node: TreeNode, result: int, self_rave_pts: list[tuple[int, int]], opp_rave_pts: list[tuple[int, int]]):
        """
        Backpropagates the simulation result through the tree.
        Args:
            node (TreeNode): The node to backpropagate from.
            result (int): The result of the simulation. (-1 for loss, 1 for win)
        """
        while node is not None:
            
            children = [(child.move.x, child.move.y) for child in node.children]
            if node.player == self.colour:
                for point in self_rave_pts:
                    if point in children:
                        i = children.index(point)
                        node.children[i].q_rave += -result
                        node.children[i].n_rave += 1
            else:
                for point in opp_rave_pts:
                    if point in children:
                        i = children.index(point)
                        node.children[i].q_rave += -result
                        node.children[i].n_rave += 1

            node.visits += 1
            node.wins += result
            result = -result
            node = node.parent

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

class MCTSAgent(AgentBase):
    """An agent that uses MCTS for Hex."""
    logger = logging.getLogger(__name__)

    def __init__(self, colour: Colour, turn_length_s: int = 1):
        super().__init__(colour)
        self.turn_length = turn_length_s # max length of a turn in seconds
        self.tree = None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Selects a move using MCTS."""
        if self.tree is None and opp_move is not None:
            logging.info("Initialising game tree...")
            self.tree = TreeNode(player=self.colour.opposite())
            self.tree = self.update_tree(self.tree, opp_move)

            x, y = opp_move.x, opp_move.y
            board.set_tile_colour(x, y, self.colour.opposite())
        elif self.tree is None:
            logging.info("Initialising game tree...")
            self.tree = TreeNode(player=self.colour)

        mcts = MCTS(board, self.colour, turn_length_s=self.turn_length)
        self.tree = mcts.run(self.tree)

        x, y = self.tree.move.x, self.tree.move.y
        board.set_tile_colour(x, y, self.colour)

        return self.tree.move
    
    def update_tree(self, tree: TreeNode, move: Move):
        """Updates the tree with the opponent's move."""
        for child in tree.children:
            if child.move == move:
                return child
        
        # If the move is not in the tree, create a new node
        return TreeNode(parent=tree, move=move)

