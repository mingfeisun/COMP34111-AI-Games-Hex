from src.Board import Board
from src.Tile import Tile
from src.Colour import Colour
from src.Move import Move

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
    
    def get_connected_strings(board: Board, colour) -> list[tuple[set[Tile], str]]:
        """
        Returns a list of named groups of connected tiles for the given colour.
        """
        connected_strings = []
        visited = set()

        for x in range(board.size):
            for y in range(board.size):
                tile = board.tiles[x][y]
                if tile.colour == colour and tile not in visited:
                    connected_string, string_type, visited = Utilities.get_connected_string(board, x, y, colour)
                    connected_strings.append((connected_string, string_type))
        
        return connected_strings
    
    def get_connected_string(board, x, y, colour):
        """
        Returns a named group of connected tiles starting from the given position.
        """
        tile = board.tiles[x][y]

        if tile.colour != colour:
            raise ValueError("Tile must be of the specified colour")
        
        neighbours = set(Utilities.get_neighbours(board, x, y))
        connected_string = ConnectedString(board.size, colour)
        connected_string.add_tile(tile)
        visited = {tile}

        while len(neighbours) > 0:
            n = neighbours.pop()
            if n not in visited and n.colour == colour:
                connected_string.add_tile(n)
                neighbours.update(Utilities.get_neighbours(board, n.x, n.y))
            visited.add(n)
                
        return connected_string.connected_tiles, connected_string.string_type, visited
    
    @staticmethod
    def find_one_to_connect(board: Board, colour: Colour) -> list[tuple[Tile, Tile]]:
        """
        Identify all pairs of groups that are one move away from connecting.
        Args:
            board (Board): The game board.
            colour (Colour): The player's colour.
        Returns:
            list[tuple[Move, ConnectedString, ConnectedString]]: A list of one-to-connect connections.
        """
        connections = []
        connected_strings = Utilities.get_connected_strings(board, colour)
        for i, connected_string_1 in enumerate(connected_strings):
            for j, connected_string_2 in enumerate(connected_strings):
                if i >= j:
                    continue

                potential_connectors = Utilities.find_potential_connectors(connected_string_1, connected_string_2, board, colour)
                for connector in potential_connectors:
                    move = Move(connector.x, connector.y)
                    connections.append((move, connected_string_1, connected_string_2))

        return connections
    
    @staticmethod
    def find_potential_connectors(connected_string_1, connected_string_2, board, colour) -> list[Tile]:
        """
        Find all potential connectors between two groups.
        """
        group_1_perimeter_tiles = Utilities.get_group_perimeter(board, connected_string_1.connected_tiles)
        group_2_perimeter_tiles = Utilities.get_group_perimeter(board, connected_string_2.connected_tiles)

        potential_connectors = group_1_perimeter_tiles.intersection(group_2_perimeter_tiles)
        return potential_connectors
    
    @staticmethod
    def get_group_perimeter(board, group) -> set[Tile]:
        """
        Returns the tiles representing the empty tiles along the perimeter of the group.
        """
        perimeter = set()
        for tile in group.connected_tiles:
            neighbours = Utilities.get_neighbours(board, tile.x, tile.y)
            for n in neighbours:
                if n.colour == None:
                    perimeter.add(n)
        
        return perimeter


######################################################################################        

class ConnectedString:
    """
    Represents a group of connected tiles on the board.
    """
    def __init__(self, board_size: int, colour: Colour):
        self.board_size = board_size
        self.colour = colour

        # The tiles in the connected string
        self.connected_tiles = set()

        # The relationship of the group with borders
        self._top_or_left_connected = False # handle both top and left
        self._bottom_or_right_connected = False # handle both bottom and right

        # The relationship of the  group to other groups
        self.group_type = ''

    def add_tile(self, tile: Tile):
        self._update_string_type(tile)
        self.connected_tiles.add(tile)

    def add_tiles(self, tiles: set[Tile]):
        self.connected_tiles.update(tiles)
    
    def _update_string_type(self, tile):
        # Looking for connections top/bottom
        if self.colour == Colour.RED:
            if tile.x == 0:
                self.top_or_left_connected = True
            elif tile.x == self.board_size - 1:
                self.bottom_or_right_connected = True
        # Looking for connections left/right
        else:
            if tile.y == 0:
                self.top_or_left_connected = True
            elif tile.y == self.board_size - 1:
                self.bottom_or_right_connected = True
    
    @property
    def string_type(self) -> str:
        string_type = ''
        if self.colour == Colour.RED:
            if self.top_or_left_connected:
                string_type += 'Top'
            if self.bottom_or_right_connected:
                string_type += 'Bottom'
        else:
            if self.top_or_left_connected:
                string_type += 'Left'
            if self.bottom_or_right_connected:
                string_type += 'Right'
        return string_type