import sys
import os

# Add project root to import path so "src" becomes visible
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


import numpy as np
from src.Board import Board
from src.Move import Move
from src.Colour import Colour


class BoardStateNP:
    """
    A NumPy-based wrapper around the engine's Board class.
    Allows faster evaluation and feature extraction for your agent.
    """

    def __init__(self, board):
        self.size = board.size
        self.array = self._convert(board)

    def _convert(self, board):

        tiles = board.tiles
        arr = np.zeros((self.size, self.size), dtype=np.int8)

        for x in range(self.size):
            for y in range(self.size):
                colour = tiles[x][y].colour

                if colour == Colour.RED:
                    arr[x, y] = 1
                elif colour == Colour.BLUE:
                    arr[x, y] = 2
                else:
                    arr[x, y] = 0  # empty
        return arr
    
    def get_numpy(self):
        return self.array

    def clone(self):
        new = BoardStateNP.__new__(BoardStateNP)
        new.size = self.size
        new.array = self.array.copy()
        return new

    def is_legal(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size and self.array[x, y] == 0

    def apply_move(self, move, colour):
        if move.is_swap():
            return  # swap does NOT alter the board

        x, y = move.x, move.y   

        if colour == Colour.RED:
            self.array[x, y] = 1
        elif colour == Colour.BLUE:
            self.array[x, y] = 2
    
    def get_neighbours(self, x, y):
        """
        Returns a list of valid neighbouring coordinates for Hex.
        Hex neighbours (6 directions):
            (x-1, y)
            (x+1, y)
            (x,   y-1)
            (x,   y+1)
            (x-1, y+1)
            (x+1, y-1)
        """

        candidates = [
            (x - 1, y),     # left
            (x + 1, y),     # right
            (x,     y - 1), # up
            (x,     y + 1), # down
            (x - 1, y + 1), # diag DL
            (x + 1, y - 1), # diag UR
        ]

        # Keep only inside-board coordinates
        valid = [
            (nx, ny)
            for (nx, ny) in candidates
            if 0 <= nx < self.size and 0 <= ny < self.size
        ]

        return valid
