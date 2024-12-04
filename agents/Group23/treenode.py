import math
from copy import deepcopy

from agents.Group23.chain import Chain
from agents.Group23.utilities import Utilities
from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile

class TreeNode:
    """Represents a node in the MCTS tree."""

    def __init__(self, board, player, move=None, parent=None):
        self.board = board  # The board state at this node
        self.move = move  # The move that led to this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.player = player  # The player to move at this node

        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node

        self.chains = TreeNode.find_connected_chains(self.board, self.player)
        self.one_to_connect_moves = TreeNode.find_one_to_connect_moves(self.board, self.player, self.chains)
        self.one_possible_connect_moves = TreeNode.find_one_possible_connect_moves(self.board, self.player, self.chains)

    def is_fully_expanded(self, legal_moves):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_param=math.sqrt(2)):
        """Selects the best child using UCT."""
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits) + exploration_param * math.sqrt(math.log(self.visits) / child.visits)
        )

    def add_child(self, move):
        """Adds a child node for a move."""
        new_board = deepcopy(self.board)
        new_board.set_tile_colour(move.x, move.y, self.player)
        child_node = TreeNode(board=new_board, 
                              player=self.player.opposite(), 
                              move=move, 
                              parent=self)
        self.children.append(child_node)
        return child_node
    
    def get_child(self, move):
        """Gets the child node for a move."""
        for child in self.children:
            if child.move == move:
                return child
        return None
    

######################################################################################
#                                   STATIC METHODS                                   #
######################################################################################
    

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
    def find_one_to_connect_moves(board: Board, colour: Colour, chains: set[Chain]) -> set[tuple[int, int]]:
        """
        Finds all moves that connect a chain to the edge of the board or to another chain.

        """
        moves = set()

        for chain in chains:
            influence_region = chain.get_influence_region(board)
            for pos in influence_region:
                # Check if this move creates a new connection (e.g., joins chains or completes a path)
                connected_chains = [
                    other_chain for other_chain in chains 
                    if other_chain != chain 
                    and pos in other_chain.get_influence_region(board)
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

    @staticmethod
    def find_one_possible_connect_moves(board: Board, colour: Colour, chains: set[Chain]) -> set[tuple[int, int]]:
        moves = set()

        for chain in chains:
            influence_region = chain.get_influence_region(board)
            for pos in influence_region:
                neighbour_tiles = Utilities.get_neighbours(board, board.tiles[pos[0]][pos[1]])
                neighbour_positions = [(tile.x, tile.y) for tile in neighbour_tiles] # tiles are unhashable
                possible_connected_chains = [
                    other_chain for other_chain in chains 
                    if other_chain != chain 
                    and len(other_chain.get_influence_region(board).intersection(neighbour_positions)) > 0
                ]
                if len(possible_connected_chains) > 0:
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
    