from src.Colour import Colour

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

    def add_tile(self, position: tuple[int, int]):
        if position[0] == 0 or position[1] == 0:
            self.is_nw_edge = True
        if position[0] == self.size - 1 or position[1] == self.size - 1:
            self.is_se_edge = True

        self.tiles.add(position)
    
    def add_tiles(self, positions: set[tuple[int, int]]):
        self.tiles |= positions
    
    def merge_chains(self, chain):
        self.tiles |= chain.tiles
        self.is_nw_edge = self.is_nw_edge or chain.is_nw_edge
        self.is_se_edge = self.is_se_edge or chain.is_se_edge

    @property
    def chain_type(self) -> int:
        if self.is_nw_edge and self.is_se_edge:
            return 'TopBottom' if self.colour == Colour.RED else 'LeftRight'
        if self.is_nw_edge:
            return 'Top' if self.colour == Colour.RED else 'Left'
        if self.is_se_edge:
            return 'Bottom' if self.colour == Colour.RED else 'Right'
        return 'Misc'