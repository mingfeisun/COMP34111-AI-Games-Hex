from src.Board import Board
from src.Tile import Tile

class Utilities:
    @staticmethod
    def is_on_board(board: Board, x: int, y: int) -> bool:
        """Returns True if the coordinates are on the board."""
        return x >= 0 and x < board.size and y >= 0 and y < board.size
    
    @staticmethod
    def get_neighbours(board: Board, tile: Tile) -> list[Tile]:
        """Returns a list of the neighbours of a tile."""
        neighbours = []
        for idx in range(Tile.NEIGHBOUR_COUNT):
            x_n = tile.x + Tile.I_DISPLACEMENTS[idx]
            y_n = tile.y + Tile.J_DISPLACEMENTS[idx]
            if Utilities.is_on_board(board, x_n, y_n):
                tile = board.tiles[x_n][y_n]
                neighbours.append(tile)
        return neighbours
    
    @staticmethod
    def clear_visits(board: Board) -> None:
        """Clears the visited status of all tiles on the board."""
        for line in board.tiles:
            for tile in line:
                tile.clear_visit()