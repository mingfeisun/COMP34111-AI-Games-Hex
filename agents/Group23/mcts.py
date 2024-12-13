from copy import deepcopy
import random
import time
import os

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from multiprocessing import Pool

from agents.Group23.treenode import TreeNode

class RolloutPolicy:
    DEFAULT_POLICY = "default_policy"
    BRIDGE_ROLLOUT_POLICY = "bridge_rollout_policy"

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5, custom_trained_network=None):
        self.colour = colour  # Agent's colour
        self.max_simulation_length = max_simulation_length  # Length of a MCTS search in seconds
        self.rollout_policy = RolloutPolicy.BRIDGE_ROLLOUT_POLICY

    def run_simulation_with_process(self, root: TreeNode, colour: Colour,start_time: float) -> int:
        """Runs a simulation with a new process."""

        iterations = 0

        while time.time() - start_time < self.max_simulation_length:
            node = self._select(root)
            result = self._simulate(node.board, colour)
            self._backpropagate(node, result)
            iterations += 1

        # delete second children
        for child in root.children:
            for grandchild in child.children:
                del grandchild

        return root, iterations

    def run(self, root: TreeNode):
        """Performs MCTS simulations from the root node."""

        _root = deepcopy(root)

        iterations = 0
        start_time = time.time()
        number_of_workers = os.cpu_count()

        if number_of_workers is None:
            number_of_workers = 8

        # result = self._simulate(node.board, self.colour)
        with Pool(number_of_workers) as p:
            results = p.starmap(self.run_simulation_with_process, [(_root, self.colour, start_time)] * number_of_workers)
        for result in results:
            iterations += result[1]
            for child in result[0].children:
                if child not in _root.children:
                    _root.children.append(child)
                else:
                    _root.children[_root.children.index(child)].visits += child.visits
                    _root.children[_root.children.index(child)].wins += child.wins

        finish_time = time.time()
        print(f'Ran {iterations} simulations in {finish_time - start_time:.2f}s')

        # Choose the most visited child as the best move if visits > 0, otherwise -inf
        best_child = max(_root.children, key=lambda child: child.wins / child.visits if child.visits > 0 else float('-inf'))
        best_child.parent = None # Remove the parent reference to reduce memory overhead
        
        return best_child, None

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
            # Move based on rollout policy
            moves = self.get_all_moves(board)
            if self.rollout_policy == RolloutPolicy.DEFAULT_POLICY:
                move = self._default_policy(moves)
            elif self.rollout_policy == RolloutPolicy.BRIDGE_ROLLOUT_POLICY:
                move = self._bridge_policy(board, colour, moves)

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
        print('======================================')
        print(f'Retrieved {len(moves)} moves from the node.')
        print(f'Moves:')
        for move in moves:
            print(f' - ({move.x}, {move.y}): priority {move.priority}')
        
        moves = [Move(move.x, move.y) for move in moves]

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
    
    def _bridge_policy(self, board: Board, colour, moves: list[Move]) -> Move:
        """
        Implements the bridge rollout pattern to select a simulation move.

        If the opponent's move probes any of the player's bridges, then the player always responds by making a bridge connection.
        Otherwise, a random move is returned.
        Resource: https://webdocs.cs.ualberta.ca/~hayward/papers/mcts-hex.pdf (IV C.) 
        """
        if len(moves) == 0:
            raise ValueError("No legal moves available.")
        
        # Search for bridge-connect patterns
        oppositeColour = Colour.opposite(colour)

        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour:
                        # top to bottom bridges
                        if i - 1 >= 0 and j + 2 < board.size and board.tiles[i-1][j+2].colour == colour:
                            # check to see if intermediate cells are taken by opponent
                            if board.tiles[i-1][j+1].colour == oppositeColour and board.tiles[i][j+1].colour == None:
                                return Move(i, j+1)     # we take the empty cell
                            elif board.tiles[i][j+1].colour == oppositeColour and board.tiles[i-1][j+1].colour == None:
                                return Move(i-1, j+1)

                        # top-right to bottom-left bridges
                        if i - 2 >= 0 and j + 1 < board.size and board.tiles[i-2][j+1].colour == colour:
                            # check to see if intermediate cells are taken by opponent
                            if board.tiles[i-1][j].colour == oppositeColour and board.tiles[i-1][j+1].colour == None:
                                return Move(i-1, j+1)
                            elif board.tiles[i-1][j+1].colour == oppositeColour and board.tiles[i-1][j].colour == None:
                                return Move(i-1, j)
                        
                        # top-left to bottom-right bridges
                        if i + 1 < board.size and j + 1 < board.size and board.tiles[i+1][j+1].colour == colour:
                            # check to see if intermediate cells are taken by opponent
                            if board.tiles[i+1][j].colour == oppositeColour and board.tiles[i][j+1].colour == None:
                                return Move(i, j+1)
                            elif board.tiles[i][j+1].colour == oppositeColour and board.tiles[i+1][j].colour == None:
                                return Move(i+1, j)

        # Return a random move when no bridge-connect pattern is found
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