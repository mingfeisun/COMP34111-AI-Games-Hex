from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from GameTree import GameTree

class GameTree:
    _board: Board
    _board_size: int
    _valid_moves: list[Move]
    _children: list[GameTree]
    q: int # the sum of all payoffs received
    n: int # the number of visits

    def __init__(self, board: Board, board_size: int):
        self._board = board
        self._board_size = board_size
        self._children = []
        self._q = 0
        self._n = 0

        tiles = self._board.get_tiles()
        self._valid_moves = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
            if tiles[i][j].colour is None
        ]
    
    def get_node(self, board: Board) -> GameTree:
        if self._board == board:
            return self
        for child in self._children:
            node = child.get_node(board)
            if node:
                return node
        return None
    
    def add_child(self, board: Board):
        child = GameTree(board)
        self._children.append(child)
    
    def get_children(self) -> list[GameTree]:
        return self._children
    
    def get_board(self) -> Board:
        return self._board
    
    def get_valid_moves(self) -> list[Move]:
        return self._valid_moves