from copy import deepcopy
import random
import time
import numpy as np

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import numpy as np

from agents.Group23.alpha_zero_treenode import TreeNode

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5, custom_trained_network=None, in_training=False):
        self.colour = colour  # Agent's colour
        self.max_simulation_length = max_simulation_length  # Length of a MCTS search in seconds
        self._trained_network = custom_trained_network
        self._agent_in_training = in_training

        if self._agent_in_training:
            self.max_depth = 60
        else:
            self.max_depth = 30

    def _get_average_value_distribution(self, node: TreeNode) -> list[list[float]]:
        """Returns the average value distribution for the children of the given node.

        Args:
            node (TreeNode): current node

        Returns:
            list[list[float]]: average value distribution
        """
        visit_distribution = np.array(self._get_visit_count_distribution(node))
        win_distribution = np.array(self._get_win_count_distribution(node))

        if visit_distribution.shape != win_distribution.shape:
            raise ValueError('Visit and win distributions have different shapes')
        
        average_value_distribution = win_distribution / visit_distribution

        return average_value_distribution

    def _get_average_value_distribution(self, node: TreeNode) -> list[list[float]]:
        """Returns the average value distribution for the children of the given node.

        Args:
            node (TreeNode): current node

        Returns:
            list[list[float]]: average value distribution
        """
        visit_distribution = np.array(self._get_visit_count_distribution(node))
        win_distribution = np.array(self._get_win_count_distribution(node))

        if visit_distribution.shape != win_distribution.shape:
            raise ValueError('Visit and win distributions have different shapes')
        
        average_value_distribution = win_distribution / visit_distribution

        return average_value_distribution

    def _get_visit_count_distribution(self, node: TreeNode) -> list[list[int]]:
        """Returns the visit count distribution for the children of the given node.

        Args:
            node (TreeNode): current node

        Returns:
            list[list[int]]: visit count distribution
        """
        distribution_board = [[0 for _ in range(11)] for _ in range(11)]
        self._count_visits_DFS(node, distribution_board)

        return distribution_board
    
    def _get_win_count_distribution(self, node: TreeNode) -> list[list[int]]:
        """Returns the win count distribution for the children of the given node.
        
        Args:
            node (TreeNode): current node

        Returns:
            list[list[int]]: win count distribution
        """
        distribution_board = [[0 for _ in range(11)] for _ in range(11)]
        self._count_wins_DFS(node, distribution_board)

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
    
    def _count_wins_DFS(self, node: TreeNode, distribution_board: list[list[int]]):
        """Counts the wins for the children of the given node.

        Args:
            node (TreeNode): current node
            distribution_board (list[list[int]]): win count distribution passed by reference
        """
        
        for child in node.children:
            x, y = child.move.x, child.move.y
            distribution_board[x][y] += child.wins
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
        best_child = max(
            root.children,
            key=lambda child: (
                (np.sum(child.wins) / np.sum(child.visits)) if np.sum(child.visits) > 0 else float('-inf')
            )
        )

        print(f'Selected move with {best_child.visits} visits and {best_child.wins} wins from {len(root.children)} possible moves')
        print(f'Moves:')
        for child in root.children:
            print(f'  - Move: ({child.move.x, child.move.y}), Wins: {child.wins}, Visits: {child.visits}')

        if self._agent_in_training:
            pd_distribution = self._get_average_value_distribution(root)
        else:
            pd_distribution = None
        
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

        current_depth = 0

        while (not simulation_board.has_ended(colour=current_colour) and
               not simulation_board.has_ended(colour=current_colour.opposite()) and
               self.max_depth < current_depth):
            
            current_depth += 1
            
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
        else:
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

        if self._trained_network is None:
            return random.choice(moves)

        # Use the neural network to select the best move
        board_vector = self.get_board_vector(self.board)
        move_distribution = self._trained_network.get_policy_value(board_vector)
        
        best_move_prob = 0
        best_move = None
        for i, in move_distribution:
            for j in move_distribution[i]:
                if move_distribution[i][j] > best_move_prob:
                    best_move_prob = move_distribution[i][j]
                    best_move = (i, j)
        
        best_move = Move(best_move[0], best_move[1])
        return best_move
