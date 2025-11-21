from .Node import Node
from src.Board import Board
from src.Colour import Colour
from random import choice

class MCTSNode(Node):
    def __init__(
        self, 
        colour: Colour,
        board: Board = None,
    ):
        self.board = board
        self.colour = colour

    def get_board(self) -> Board:
        return getattr(self, "board", None)
    
    def get_player(self) -> Colour:
        return getattr(self, "colour", None)

    def find_children(self):
        "All possible successors of this board state"
        board = self.get_board()

        if board is None:
            return set()

        children = set()
        try:
            size = board.size
            tiles = board.tiles
        except Exception:
            return set()

        for i in range(size):
            for j in range(size):
                try:
                    if tiles[i][j].colour == None:
                        children.add((i, j))
                except Exception:
                    continue
        return children
    
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        board = self.get_board()
        if board is None:
            return None
        
        try:
            size = board.size
            tiles = board.tiles
        except Exception:
            return None

        empty_tiles = []
        for i in range(size):
            for j in range(size):
                try:
                    if tiles[i][j].colour == None:
                        empty_tiles.append((i, j))
                except Exception:
                    continue

        if not empty_tiles:
            return None

        return choice(empty_tiles)
    
    def is_terminal(self):
        "Returns True if the node has no children"
        return True
    
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss"
        board = self.get_board()

        if board is None:
            return 0
        
        try:
            winner = board.get_winner()
        except Exception:
            winner = getattr(board, "_winner", None)

        # if no winner, not a terminal win for anyone
        if winner is None:
            return 0
        
        player = self.get_player()

        if player is None:
            return 0
        
        return 1 if winner == player else 0
    
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789
    
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True