from itertools import chain
from src.Board import Board
from src.Colour import Colour

from agents.Group23.utilities import Utilities

class Chain:
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4406406
    Represents a disjoint set of same-coloured connected tiles.
    """
    def __init__(self, board_size: int, colour: Colour):
        self.size = board_size
        self.colour = colour

        self.tiles = set()
        self.is_nw_edge = False
        self.is_se_edge = False

        self._influence_region = None

    def add_tile(self, position: tuple[int, int]):
        if position[0] == 0 or position[1] == 0:
            self.is_nw_edge = True
        if position[0] == self.size - 1 or position[1] == self.size - 1:
            self.is_se_edge = True

        self.tiles.add(position)

        # Clear the influence region (cache invalidation)
        self.influence_region = None
    
    def add_tiles(self, positions: set[tuple[int, int]]):
        self.tiles |= positions

        # Clear the influence region (cache invalidation)
        self.influence_region = None

    
    def merge_chains(self, chain):
        self.tiles |= chain.tiles
        self.is_nw_edge = self.is_nw_edge or chain.is_nw_edge
        self.is_se_edge = self.is_se_edge or chain.is_se_edge

        # Clear the influence region (cache invalidation)
        self.influence_region = None

    @property
    def chain_type(self) -> int:
        if self.is_nw_edge and self.is_se_edge:
            return 'TopBottom' if self.colour == Colour.RED else 'LeftRight'
        if self.is_nw_edge:
            return 'Top' if self.colour == Colour.RED else 'Left'
        if self.is_se_edge:
            return 'Bottom' if self.colour == Colour.RED else 'Right'
        return 'Misc'
    
    def get_influence_region(self, board) -> set[tuple[int, int]]:
        if self._influence_region is not None:
            # return cached value
            return self._influence_region

        self._influence_region = set()

        for (x, y) in self.tiles:
            tile = board.tiles[x][y]
            for neighbour in Utilities.get_neighbours(board, tile):
                if neighbour.colour == None:
                    self._influence_region.add((neighbour.x, neighbour.y))

        return self._influence_region