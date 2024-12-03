from agents.Group23.chain import Chain

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile
from src.Move import Move

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
    def find_connected_chains(board: Board, colour: Colour) -> set[Chain]:
        """Finds all connected chains of the given colour."""
        chains = set()

        def dfs(position: tuple[int, int], chain: Chain, colour: Colour):
            x, y = position
            board.tiles[x][y].visit()
            chain.add_tile(position)

            for idx in range(Tile.NEIGHBOUR_COUNT):
                x_n = x + Tile.I_DISPLACEMENTS[idx]
                y_n = y + Tile.J_DISPLACEMENTS[idx]
                if Utilities.is_on_board(board, x_n, y_n):
                    neighbour = board.tiles[x_n][y_n]

                    if not neighbour.is_visited() and neighbour.colour == colour:
                        dfs((x_n, y_n), chain, colour)
        
        for x in range(board.size):
            for y in range(board.size):
                tile = board.tiles[x][y]

                if not tile.is_visited() and tile.colour == colour:
                    chain = Chain(board.size, colour)
                    dfs((x, y), chain, colour)
                    chains.add(chain)
        
        Utilities.clear_visits(board)

        return chains
    
    @staticmethod
    def clear_visits(board: Board) -> None:
        """Clears the visited status of all tiles on the board."""
        for line in board.tiles:
            for tile in line:
                tile.clear_visit()

    # @staticmethod
    # def find_virtual_connections(board: Board, colour: Colour) -> set[tuple[int, int]]:
    #     """Finds all virtual connections of the given colour."""
    #     virtual_connections = set()

    #     # Find all connected chains
    #     chains = Utilities.find_connected_chains(board, colour)

    #     # Find virtual connections
    #     for chain in chains:
    #         if chain.chain_type == 'TopBottom' or chain.chain_type == 'LeftRight':
    #             virtual_connections.add(chain)
        
    #     return virtual_connections

    @staticmethod
    def find_influence_region(board: Board, chain: Chain) -> set[tuple[int, int]]:
        """Finds the influence region of a chain."""
        influence_region = set()

        for (x, y) in chain.tiles:
            tile = board.tiles[x][y]
            for neighbour in Utilities.get_neighbours(board, tile):
                if neighbour.colour == None:
                    influence_region.add((neighbour.x, neighbour.y))
        
        return influence_region
    
    @staticmethod
    def find_one_to_connect_moves(board: Board, colour: Colour) -> set[tuple[int, int]]:
        """Finds all moves that connect a chain to the edge of the board."""
        moves = set()

        chains = Utilities.find_connected_chains(board, colour)
        influence_regions = {chain: Utilities.find_influence_region(board, chain) for chain in chains}

        for chain in chains:
            for pos in influence_regions[chain]:
                # Check if this move creates a new connection (e.g., joins chains or completes a path)
                connected_chains = [
                    other_chain for other_chain in chains 
                    if other_chain != chain and pos in influence_regions[other_chain]
                ]
                if len(connected_chains) > 0:
                    moves.add(pos)
                    
                # Check if this move connects the chain to a new edge
                if colour == Colour.RED:
                    if 'Top' not in chain.chain_type and pos[0] == 0:
                        moves.add(pos)
                    if 'Bottom' not in chain.chain_type and pos[0] == board.size - 1:
                        moves.add(pos)
                elif colour == Colour.BLUE:
                    if 'Left' not in chain.chain_type and pos[1] == 0:
                        moves.add(pos)
                    if 'Right' not in chain.chain_type and pos[1] == board.size - 1:
                        moves.add(pos)
        
        return moves

