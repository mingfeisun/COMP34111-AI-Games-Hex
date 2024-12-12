from copy import deepcopy
import random
import time

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.Group23.treenode import TreeNode

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5, custom_trained_network=None):
        self.colour = colour  # Agent's colour
        self.max_simulation_length = max_simulation_length  # Length of a MCTS search in seconds

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


    def run(self, root: TreeNode):
        """Performs MCTS simulations from the root node."""
        iterations = 0
        start_time = time.time()
        while time.time() - start_time < self.max_simulation_length:
            iterations += 1
            node = self._select(root)
            result = self._simulate(node.board, self.colour)
            self._backpropagate(node, result)

        finish_time = time.time()
        print(f'Ran {iterations} simulations in {finish_time - start_time:.2f}s')

        # Choose the most visited child as the best move
        best_child = max(root.children, key=lambda child: child.wins / child.visits)
        best_child.parent = None # Remove the parent reference to reduce memory overhead

        pd_distribution = self._get_visit_count_distribution(root)
        
        return best_child, pd_distribution

    def _select(self, node: TreeNode):
        """Selects a node to expand using the UCT formula."""
        moves = self.get_heuristic_moves(node)
        while node.is_fully_expanded(moves):
            node = node.best_child()
        return self._expand(node)

    def _expand(self, node: TreeNode):
        """Expands the node by adding a new child."""
        moves = self.get_heuristic_moves(node)
        unvisited_moves = [move for move in moves if (move.x, move.y) not in [(child.move.x, child.move.y) for child in node.children]]

        if len(unvisited_moves) > 0:
            new_move = random.choice(unvisited_moves)
            return node.add_child(new_move)

        return node

    def _simulate(self, board: Board, colour: Colour):
        """Simulates a random game from the current node and returns the result."""
        # Stores the visited moves for backpropagation
        player = colour
        board = deepcopy(board)

        # Play randomly until the game ends
        while (not board.has_ended(colour=colour) and
               not board.has_ended(colour=colour.opposite())):
            moves = self.get_all_moves(board)

            move = self._default_policy(moves)

            # use tuple of coordinates for speed
            x, y = move.x, move.y

            board.set_tile_colour(x, y, colour)
            
            colour = colour.opposite()

        result = 1 if board.get_winner() == player else 0
        return result

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

            moves = self._removeInferiorCells(moves, node.board)

        return moves

    def _default_policy(self, moves: list[Move]) -> Move:
        """
        Implements a default policy to select a simulation move.
        """
        if len(moves) == 0:
            raise ValueError("No legal moves available")
        return random.choice(moves)
    
    def _removeInferiorCells(self, moves, board):

        movesOut = [move for move in moves if not self._patternMatch(move, board)]
        return movesOut
    
    
    offsets = [(0,-1),(1,-1),(1,0),(0,1),(-1,1),(-1,0)] #neighbour offsets
    colours = [Colour.RED,Colour.BLUE]
    patterns = [["A","A","C1","C1","C1","C1"],["A","C1","C1","A","C2","C2"],["C1","C1","C1","A","C2","A"]]

    def _patternMatch(self, move, board):
        #boardActual = board
        board = board._tiles
        if board[move._x][move._y]._colour != None:
            return False

        for pattern in self.patterns:
            for colour in self.colours:
                for i in range(6): #try all 6 rotations of pattern
                    rotated = pattern[i:] + pattern[:i]
                    count = 0
                    for j in range(6): #match each pattern entry to relevant neighbour tile
                        #check out of bounds and assign correct colour
                        if move._x+self.offsets[j][0] < 0 or move._x+self.offsets[j][0] > len(board[0])-1:
                            tile = Colour.BLUE
                        elif move._y+self.offsets[j][1] < 0 or move._y+self.offsets[j][1] > len(board[0])-1:
                            tile = Colour.RED
                        else:
                            tile = board[move._x+self.offsets[j][0]][move._y+self.offsets[j][1]]._colour
                        
                        #check if neighbour tile is correct colour to match the pattern
                        if rotated[j] == "A":
                            count += 1
                            continue #irrelevant tile
                        if rotated[j] == "C1" and  tile == colour:
                            count += 1
                        if rotated[j] == "C2" and tile != colour and tile != None:
                            count += 1                            
                    if count == 6:
                        #every neighbour tile has matched the pattern so this move is a dead cell
                        #print("=======================")
                        #print(boardActual.print_board())
                        #print(board[move._x][move._y]._colour)
                        #print(move)
                        return True
        return False