import random

from src.Board import Board
from src.Colour import Colour

class ZobristHasher:
    def __init__(self, size):
        self.size = size
        self.zobrist_table = {
            (row, col, colour): random.getrandbits(64)
            for row in range(size)
            for col in range(size)
            for colour in (Colour.BLUE, Colour.RED, None)
        }
        self.empty_tile = random.getrandbits(64)

        self._init_transposition_table()
    
    def hash(self, board: Board):
        hash = 0
        for row in range(self.size):
            for col in range(self.size):
                tile = board.tiles[row][col]
                hash ^= self.zobrist_table[(row, col, tile.colour)]
        return hash
    
    def _init_transposition_table(self):
        """
        Initializes the transposition table for the early game. 
        Assumes an 11x11 board.

        Returns:
            dict[int, move]: Transposition table. The key is the hash of the board state, and the value is the best move.
        """
        self.transposition_table = {}
        self.transposition_table.update(self.get_a2_b2_c2_openings())
        self.transposition_table.update(self.get_d2_openings())
        self.transposition_table.update(self.get_a6_openings())
        self.transposition_table.update(self.get_a7_openings())
        self.transposition_table.update(self.get_a8_openings())
        self.transposition_table.update(self.get_a9_openings())
        self.transposition_table.update(self.get_a11_openings())
        self.transposition_table.update(self.get_f3_openings())
    
    def get_move(self, hash):
        """
        Returns the best move for the given hash.

        Args:
            hash (int): The hash of the board state.

        Returns:
            Move: The best move for the given hash.
        """
        return self.transposition_table.get(hash, None)

        
    def get_a2_b2_c2_openings(self):
        """
        Returns the openings for the A2, B2, and C2 moves.
        """
        openings = {}

        # A2 Opening
        board = Board(11)
        # Variation 1
        board.set_tile_colour(1, 0, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 3)

        board.set_tile_colour(7, 3, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 7)

        board.set_tile_colour(6, 7, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        # Variation 2
        board.set_tile_colour(6, 7, None) # undo
        board.set_tile_colour(7, 1, Colour.RED)
        openings[hash] = (3, 7)

        board.set_tile_colour(3, 7, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (3, 9)

        # B2 Opening
        board = Board(11)
        # Variation 1
        board.set_tile_colour(1, 1, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 3)

        board.set_tile_colour(7, 3, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 7)

        board.set_tile_colour(6, 7, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        # Variation 2
        board.set_tile_colour(6, 7, None) # undo
        board.set_tile_colour(7, 1, Colour.RED)
        openings[hash] = (3, 7)

        board.set_tile_colour(3, 7, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (3, 9)


        # C2 Opening
        board = Board(11)
        # Variation 1
        board.set_tile_colour(1, 2, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 3)

        board.set_tile_colour(7, 3, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 7)

        board.set_tile_colour(6, 7, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        # Variation 2
        board.set_tile_colour(6, 7, None) # undo
        board.set_tile_colour(7, 1, Colour.RED)
        openings[hash] = (3, 7)

        board.set_tile_colour(3, 7, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (3, 9)

        return openings

    def get_d2_openings(self):
        """
        Returns the opening for the D2 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(1, 3, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 3)

        return openings

    def get_a6_openings(self):
        """
        Returns the opening for the A6 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(5, 0, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 3)

        board.set_tile_colour(7, 3, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 7)

        board.set_tile_colour(6, 7, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        board.set_tile_colour(3, 7, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 1)

        board.set_tile_colour(6, 1, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (2, 3)

        return openings
    
    def get_a7_openings(self):
        """
        Returns the opening for the A7 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(6, 0, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (8, 2)

        board.set_tile_colour(8, 2, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 7)

        board.set_tile_colour(6, 7, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        board.set_tile_colour(3, 7, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (7, 1)

        board.set_tile_colour(7, 1, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (9, 2)

        return openings

    def get_a8_openings(self):
        """
        Returns the opening for the A8 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(7, 0, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (9, 1)

        return openings
    
    def get_a9_openings(self):
        """
        Returns the opening for the A9 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(8, 0, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (2, 3)

        board.set_tile_colour(2, 3, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (9, 1)

        board.set_tile_colour(9, 1, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 6)

        board.set_tile_colour(7, 6, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (2, 2)

        board.set_tile_colour(2, 2, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 2)

        board.set_tile_colour(3, 2, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        return openings
    
    def get_a11_openings(self):
        """
        Returns the opening for the A11 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(10, 0, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (3, 4)

        board.set_tile_colour(3, 4, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (3, 7)

        board.set_tile_colour(3, 7, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 6)

        board.set_tile_colour(7, 6, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 3)

        return openings
    
    def get_f3_openings(self):
        """
        Returns the opening for the F3 move.
        """
        openings = {}

        board = Board(11)

        board.set_tile_colour(2, 5, Colour.RED)
        hash = self.hash(board)
        openings[hash] = (7, 3)

        board.set_tile_colour(7, 3, Colour.BLUE)
        hash = self.hash(board)
        openings[hash] = (6, 7)

        return openings