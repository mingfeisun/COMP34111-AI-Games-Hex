import copy

from src.Move import Move
from .Node import Node
from src.Board import Board
from src.Colour import Colour
from random import choice, random


class MCTSNode():
    def __init__(
        self, 
        colour: Colour,
        board: Board,
        parent: "MCTSNode | None" = None
    ):
        self.board = board
        self.colour = colour

        self.parent = parent
        self.children: dict[Move, MCTSNode] = {} # Key is move, value is a Node
        self.Q = 0 # Total reward
        self.N = 0 # Total number of visits

        # Compute the list of legal moves once in the constructor so we don't have to do this again
        # TODO: What about swap (i.e. Move(-1, -1)?
        self._possible_moves = [
            Move(i, j)
            for i in range(self.board.size)
            for j in range(self.board.size)
            if not self.board.tiles[i][j].colour
        ]

    @property
    def is_terminal(self) -> bool:
        """Returns True if the game has ended"""
        return self.board.has_ended(Colour.opposite(self.colour))

    @property
    def is_fully_explored(self) -> bool:
        """Returns True if this all children of this node have been explored at least once"""
        return len(self.children) == len(self._possible_moves)

    @property
    def unexplored_moves(self) -> list[Move]:
        """Returns a list of unexplored moves (i.e those that don't have a node yet)"""
        return [move for move in self._possible_moves if move not in self.children]

    def make_move(self, move: Move) -> "MCTSNode":
        """Create the node for this move and store in children"""
        board_copy = copy.deepcopy(self.board) # Next state
        board_copy.set_tile_colour(move.x, move.y, self.colour)
        child = MCTSNode(Colour.opposite(self.colour), board_copy, self)
        self.children[move] = child
        return child