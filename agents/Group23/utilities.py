from src.Board import Board
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