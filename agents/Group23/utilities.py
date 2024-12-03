from src.Board import Board
from src.Tile import Tile
from src.Colour import Colour

from enum import Enum

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
    
    def get_groups(board: Board, colour) -> list[tuple[set[Tile], str]]:
        """
        Returns a list of named groups of connected tiles for the given colour.
        """
        groups = []
        visited = set()

        for x in range(board.size):
            for y in range(board.size):
                tile = board.tiles[x][y]
                if tile.colour == colour and tile not in visited:
                    group, group_name, visited = Utilities.get_group(board, x, y, colour)
                    groups.append((group, group_name))
        
        return groups
    
    def get_group(board, x, y, colour):
        """
        Returns a named group of connected tiles starting from the given position.
        """
        tile = board.tiles[x][y]

        if tile.colour != colour:
            raise ValueError("Tile must be of the specified colour")
        
        neighbours = set(Utilities.get_neighbours(board, x, y))
        group = {tile}
        visited = {tile}

        while len(neighbours) > 0:
            n = neighbours.pop()
            if n not in visited and n.colour == colour:
                group.add(n)
                neighbours.update(Utilities.get_neighbours(board, n.x, n.y))
            visited.add(n)
        
        # TODO - write logic to handle if pie rule invoked
        # Looking for connections top/bottom
        group_name = ''
        if colour == Colour.RED:
            for tile in group_name:
                if tile.x == 0:
                    group_name += 'Top'
                elif tile.x == board.size - 1:
                    group_name += 'Bottom'
        # Looking for connections left/right
        else:
            for tile in group_name:
                if tile.y == 0:
                    group_name += 'Left'
                elif tile.y == board.size - 1:
                    group_name += 'Right'

        if group_name.contains('Top') and group_name.contains('Bottom'):
            group_name = 'TopBottom'
        elif group_name.contains('Left') and group_name.contains('Right'):
            group_name = 'LeftRight'
                
        return group, group_name, visited
        
        




######################################################################################