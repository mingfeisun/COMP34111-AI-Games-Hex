from copy import deepcopy
import random
import time

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group23.alpha_zero_treenode import TreeNode

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5, custom_trained_network=None):
        self.colour = colour  # Agent's colour
        self.max_simulation_length = max_simulation_length  # Length of a MCTS search in seconds
        self._trained_network = custom_trained_network

    def _get_visit_count_distribution(self, node: TreeNode) -> list[list[int]]:
        """Returns the visit count distribution for the children of the given node.

        Args:
            node (TreeNode): current node

        Returns:
            list[list[int]]: visit count distribution
        """
        distribution_board = [[0 for _ in range(11)] for _ in range(11)]
        self._count_visits_DFS(node, distribution_board)

        # softmax normalization
        total_visits = sum(sum(row) for row in distribution_board)
        for i in range(11):
            for j in range(11):
                distribution_board[i][j] /= total_visits

        return distribution_board
    
    def _count_visits_DFS(self, node: TreeNode, distribution_board: list[list[int]]):
        """Counts the visits for the children of the given node.

        Args:
            node (TreeNode): current node
            distribution_board (list[list[int]]): visit count distribution passed by reference
        """
        for child in node.children:
            x, y = child.move.x, child.move.y
            distribution_board[x][y] += child.visits
            self._count_visits_DFS(child, distribution_board)


    def run(self, board: Board):
        """Performs MCTS simulations from the root node."""
        root = TreeNode(board=board, player=self.colour, trained_network=self._trained_network)

        iterations = 0
        start_time = time.time()
        while time.time() - start_time < self.max_simulation_length:
            iterations += 1
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)

        print(f'Ran {iterations} simulations in {time.time() - start_time:.2f}s')

        # Choose the most visited child as the best move
        best_child = max(root.children, key=lambda child: child.wins / child.visits)

        print(f'Selected move with {best_child.visits} visits and {best_child.wins} wins from {len(root.children)} possible moves')
        print(f'Moves:')
        for child in root.children:
            print(f'  - Move: ({child.move.x, child.move.y}), Wins: {child.wins}, Visits: {child.visits}')

        pd_distribution = self._get_visit_count_distribution(root)
        
        return best_child.move, pd_distribution

    def _select(self, node: TreeNode):
        """Selects a node to expand using the UCT formula."""
        moves = self.get_heuristic_moves(node)
        while node.is_fully_expanded(moves):
            node = node.best_child()
        return self._expand(node)

    def _expand(self, node: TreeNode):
        """Expands the node by adding a new child."""
        moves = self.get_heuristic_moves(node)
        unvisited_moves = [move for move in moves if move not in [child.move for child in node.children]]

        if len(unvisited_moves) > 0:
            new_move = random.choice(unvisited_moves)
            return node.add_child(new_move)

        return node
    
    def get_board_vector(self, board: Board) -> list[list[int]]:
        """generate input vector for neural network
        based on current and recent board states

        Args:
            board (Board): current board state

        Returns:
            list[int]: input vector for neural network
        """
        
        # convert board state to input vector
        board_vector = []
        for i in range(len(board.tiles)):
            new_line = []
            for j in range(len(board.tiles)):
                tile = board.tiles[i][j].colour
                if tile == None:
                    new_line.append(0)
                elif tile == self.colour:
                    new_line.append(1)
                else:
                    new_line.append(-1)
            board_vector.append(new_line)

        return board_vector

    def _simulate(self, node: TreeNode):
        """Simulates a random game from the current node and returns the result."""
        simulation_board = deepcopy(node.board)

        # Play randomly until the game ends
        current_colour = self.colour.opposite()

        while (not simulation_board.has_ended(colour=current_colour) and
               not simulation_board.has_ended(colour=current_colour.opposite())):
            
            moves = self.get_all_moves(simulation_board)

            move = self._default_policy(moves)

            x, y = move.x, move.y
            simulation_board.set_tile_colour(x, y, current_colour)
            current_colour = current_colour.opposite()

        simulation_board_vector = self.get_board_vector(simulation_board)

        if simulation_board.has_ended(colour=self.colour):
            value = 1
        elif simulation_board.has_ended(colour=self.colour.opposite()):
            value = 0

        # use predicted value from neural network
        value = self._trained_network.get_predicted_value(simulation_board_vector)
        return value

    def _backpropagate(self, node: TreeNode, result: int):
        """Backpropagates the simulation result through the tree."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # Invert the result for the opponent's perspective

    def get_all_moves(self, board: Board) -> list[Move]:
        choices = [
            (i, j) for i in range(board.size) for j in range(board.size)
        ]
        return [Move(x, y) for x, y in choices if board.tiles[x][y].colour == None]
    
    def get_heuristic_moves(self, node: TreeNode) -> list[Move]:
        """
        Generates a subset of all legal moves for the current board state, based on the heuristic given:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4406406
        """
        moves = node.moves

        if len(moves) == 0:
            moves = self.get_all_moves(node.board)

        return moves

    def _default_policy(self, moves: list[Move]) -> Move:
        """
        Implements a default policy to select a simulation move.
        """



        if len(moves) == 0:
            raise ValueError("No legal moves available")
        return random.choice(moves)
    
# from copy import deepcopy
# import random
# import time
# import itertools

# from src.Board import Board
# from src.Colour import Colour
# from src.Move import Move

# from agents.Group23.utilities import Utilities
# from agents.Group23.treenode import TreeNode
# from agents.Group23.Alpha_Zero_NN import Alpha_Zero_NN

# class MCTS:
#     """Implements the Monte Carlo Tree Search algorithm."""
#     _trained_network = None

#     def __init__(self, board: Board, colour: Colour, turn_length_s: int = 5, custom_trained_network:Alpha_Zero_NN=None):
#         self.board = board  # The game board
#         self.colour = colour  # Agent's colour
#         self.turn_length = turn_length_s  # Length of a MCTS search in seconds
#         self.explore = 0.4 # Exploration parameter

#         self._trained_network = custom_trained_network
#         if self._trained_network is None:
#             raise ValueError("A trained network must be provided for AlphaZero.")

        

#     def _get_visit_count_distribution(self, node: TreeNode) -> list[list[int]]:
#         """Returns the visit count distribution for the children of the given node.

#         Args:
#             node (TreeNode): current node

#         Returns:
#             list[list[int]]: visit count distribution
#         """
#         distribution_board = [[0 for _ in range(11)] for _ in range(11)]
#         self._count_visits_DFS(node, distribution_board)

#         # softmax normalization
#         total_visits = sum(sum(row) for row in distribution_board)
#         for i in range(11):
#             for j in range(11):
#                 distribution_board[i][j] /= total_visits
#         return distribution_board
    
#     def _count_visits_DFS(self, node: TreeNode, distribution_board: list[list[int]]):
#         """Counts the visits for the children of the given node.

#         Args:
#             node (TreeNode): current node
#             distribution_board (list[list[int]]): visit count distribution passed by reference
#         """
#         for child in node.children:
#             x, y = child.move.x, child.move.y
#             distribution_board[x][y] += child.visits
#             self._count_visits_DFS(child, distribution_board)


#     def run(self, root: TreeNode):
#         """Performs MCTS simulations from the root node."""
#         iterations = 0
#         start_time = time.time()
#         while time.time() - start_time < self.turn_length:
#             iterations += 1
#             node = self._select(root)
#             result = self._simulate(node)
#             self._backpropagate(node, result)

#         # Choose the most visited child as the best move
#         best_child = max(root.children, key=lambda child: child.visits)

#         visit_count_normalised_distribution = self._get_visit_count_distribution(root)
#         # print(visit_count_normalised_distribution)
#         return best_child, visit_count_normalised_distribution

#     def _select(self, node: TreeNode):
#         """Selects a node to expand using the UCT formula considering policy head"""
#         legal_moves = self._get_legal_moves(self.board)
#         while node.is_fully_expanded(legal_moves):
#             node = node.best_child()
#         return self._expand(node)

#     def _expand(self, node: TreeNode):
#         """Expands the node by adding a new child."""
#         legal_moves = self._get_legal_moves(self.board)
#         unvisited_moves = [move for move in legal_moves if move not in [child.move for child in node.children]]

#         if len(unvisited_moves) > 0:
#             new_move = random.choice(unvisited_moves)
#             return node.add_child(new_move)

#         return node
    
#     def get_board_vector(self, board: Board) -> list[list[int]]:
#         """generate input vector for neural network
#         based on current and recent board states

#         Args:
#             board (Board): current board state

#         Returns:
#             list[int]: input vector for neural network
#         """
        
#         # convert board state to input vector
#         board_vector = []
#         for i in range(len(board.tiles)):
#             new_line = []
#             for j in range(len(board.tiles)):
#                 tile = board.tiles[i][j].colour
#                 if tile == None:
#                     new_line.append(0)
#                 elif tile == self.colour:
#                     new_line.append(1)
#                 else:
#                     new_line.append(-1)
#             board_vector.append(new_line)

#         return board_vector

#     def _simulate(self, node: TreeNode):
#         """Simulates a random game from the current node and returns the result."""
#         simulation_board = deepcopy(self.board)  # Create a copy of the board
#         x, y = node.move.x, node.move.y
#         simulation_board.set_tile_colour(x, y, self.colour)  # Play the move

#         simulation_board_vector = self.get_board_vector(simulation_board)

#         # Play randomly until the game ends
#         current_colour = self.colour.opposite()
#         while (not simulation_board.has_ended(colour=current_colour) and
#                not simulation_board.has_ended(colour=current_colour.opposite())):
#             legal_moves = self._get_legal_moves(simulation_board)

#             move = self._default_policy(simulation_board, current_colour, legal_moves)

#             x, y = move.x, move.y
#             simulation_board.set_tile_colour(x, y, current_colour)
#             current_colour = current_colour.opposite()

#         value = self._trained_network.get_predicted_value(simulation_board_vector)

#         # return 1 if simulation_board.get_winner() == self.colour else 0
#         return value

#     def _backpropagate(self, node: TreeNode, result: int):
#         """Backpropagates the simulation result through the tree."""
#         while node is not None:
#             node.visits += 1
#             node.wins += result
#             node = node.parent
#             result = 1 - result  # Invert the result for the opponent's perspective

#     def _get_legal_moves(self, board: Board) -> list[Move]:
#         available_tiles = []
#         for row in board.tiles:
#             for tile in row:
#                 if tile.colour != Colour.RED and tile.colour != Colour.BLUE:
#                     available_tiles.append(tile)
        
#         if len(available_tiles) == 0:
#             raise ValueError("No legal moves available")
        
#         return [Move(tile.x, tile.y) for tile in available_tiles]
    
#     def _trained_model_policy_best_move(self, board: Board, legal_moves: list[Move]) -> list[list[float]]:
#         """
#         Returns the policy predicted by the trained model for the given board state.
        
#         Args:
#             board (Board): The current state of the board.
        
#         Returns:
#             list[list[float]]: The policy predicted by the trained model.
#         """
#         board_vector = self.get_board_vector(board)
#         policy_matrix = self._trained_network.get_policy_value(board_vector)

#         # Filter the policy for legal moves
#         legal_policy = [[0 for _ in range(11)] for _ in range(11)]
#         for move in legal_moves:
#             legal_policy[move.x][move.y] = policy_matrix[move.x][move.y]

#         # select best move
#         best_move = max(legal_moves, key=lambda move: legal_policy[move.x][move.y])

#         return best_move

#     def _default_policy(self, board: Board, colour: Colour, legal_moves: list[Move]) -> Move:
#         """
#         Implements a default policy to select a simulation move.
#         Checks for basic savebridge template connections to play deterministically.
#         """
#         # savebridge_moves = self.savebridge(board, colour, legal_moves)
#         # if len(savebridge_moves) > 0:
#         #     return random.choice(savebridge_moves)

#         if random.random() < self.explore:
#             return random.choice(legal_moves)

#         return self._trained_model_policy_best_move(board, legal_moves)
    
#     def savebridge(self, board: Board, colour: Colour, legal_moves: list[Move]) -> Move:
#         """
#         Implements the savebridge function to identify and play moves
#         that secure connections using the bridge and wheel templates.
#         Based on the template connection strategy defined in:
#         https://webdocs.cs.ualberta.ca/~mmueller/ps/2013/2013-CG-Mohex2.0.pdf

#         Args:
#             board (Board): The current state of the board.
#             colour (Colour): The player's colour.
#             legal_moves (list[Move]): A list of legal moves.

#         Returns:
#             Move | None: The move to secure a template connection, or None if no template applies.
#         """
#         # Check for bridge opportunities
#         savebridge_moves = []
#         for move in legal_moves:
#             x, y = move.x, move.y
            
#             # Look for the bridge template
#             neighbors = Utilities.get_neighbours(board, x, y)
#             for n1, n2 in itertools.combinations(neighbors, 2):
#                 # Check if n1 and n2 form a bridge with the current move
#                 if (n1.colour == colour and
#                     n2.colour == colour and
#                     self.is_bridge(n1, n2, move)):
#                     savebridge_moves.append(move)
                
#         # Check for wheel opportunities
#         for move in legal_moves:
#             x, y = move.x, move.y
            
#             # Check if the move completes a wheel template
#             if self.completes_wheel(board, x, y, colour):
#                 savebridge_moves.append(move)
        
#         return savebridge_moves
    
#     def is_bridge(self, n1, n2, move):
#         """
#         Checks if n1 and n2, with the given move, form a bridge.
        
#         Args:
#             n1 (Tile): The first neighbor.
#             n2 (Tile): The second neighbor.
#             move (Move): The move being evaluated.
        
#         Returns:
#             bool: True if the pattern forms a bridge, False otherwise.
#         """
#         # Bridge condition: Diagonal adjacency with one gap
#         return (abs(n1.x - n2.x) == 2 and abs(n1.y - n2.y) == 2 and
#                 (move.x, move.y) == ((n1.x + n2.x) // 2, (n1.y + n2.y) // 2))
    
#     def completes_wheel(self, board: Board, x: int, y: int, colour: Colour) -> bool:
#         """
#         Checks if placing a tile at (x, y) completes a wheel template.
        
#         Args:
#             board (Board): The game board.
#             x (int): X-coordinate of the move.
#             y (int): Y-coordinate of the move.
#             colour (Colour): The player's colour.
        
#         Returns:
#             bool: True if the move completes a wheel, False otherwise.
#         """
#         neighbors = Utilities.get_neighbours(board, x, y)
#         same_coloured_neighbors = [
#             neighbour for neighbour in neighbors
#             if neighbour.colour == colour
#         ]
        
#         # Wheel condition: At least 3 neighbors of the same colour
#         return len(same_coloured_neighbors) >= 3




# ######################################################################################