import math
from copy import deepcopy
import itertools
import numpy as np
from agents.Group23.chain import Chain
from agents.Group23.utilities import Utilities
from agents.Group23.heuristic_move import HeuristicMove
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile

@staticmethod
def _board_tiles_to_compact_array(board: Board, size:int) -> list[int]:
    """Converts the board tiles to a compact array."""

    tiles = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            tiles[i][j] = 1 if board.tiles[i][j].colour == Colour.RED else 2 if board.tiles[i][j].colour == Colour.BLUE else 0

    return tiles.flatten().tolist()

@staticmethod
def _compact_array_to_board_tiles(compact_array: list[int], size: int) -> list[list[Tile]]:
    """Converts a compact array to board tiles."""
    tiles = []
    for i in range(size):
        new_line = []
        for j in range(size):
            value = compact_array[i * size + j]
            if value == 1:
                new_line.append(Tile(i, j, Colour.RED))
            elif value == 2:
                new_line.append(Tile(i, j, Colour.BLUE))
            else:
                new_line.append(Tile(i, j))
        tiles.append(new_line)
    return tiles

class TreeNode:
    """Represents a node in the MCTS tree."""
    
    @property
    def board(self) -> Board:
        b = Board(self.board_size)
        b._tiles = _compact_array_to_board_tiles(self._board, self.board_size)
        return b

    def __init__(self, board, player, move=None, parent=None):
        self.board_size = board._size
        self._board = _board_tiles_to_compact_array(board, self.board_size)
        
        self.move = move  # The move that led to this node
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.player = player  # The player to move at this node

        # ucb values
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node

        self.chains = TreeNode.find_connected_chains(self.board, self.player)
        one_to_connect_moves = TreeNode.find_one_to_connect_moves(self.board, self.player, self.chains)
        one_possible_connect_moves = TreeNode.find_one_possible_connect_moves(self.board, self.player, self.chains)
        blocking_moves = TreeNode.find_blockable_moves(self.board, self.player.opposite(), self.chains)
        
        self._moves = one_to_connect_moves | one_possible_connect_moves | blocking_moves

    def is_fully_expanded(self, legal_moves):
        """Checks if all possible moves have been expanded."""
        return len(self.children) == len(legal_moves)

    def best_child(self, exploration_param=0.7):
        """Selects the best child using UCT."""
        return max(
            self.children,
            key=lambda child: child.uct_value(exploration_param)
        )
    
    def uct_value(self, exploration_param=0.7):
        """Calculates the UCT value for a node."""
        return (self.wins / self.visits) + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def add_child(self, move):
        """
        Adds a child node for a move.
        Returns the child if it already exists.
        """
        # Check if the child already exists
        for child in self.children:
            if child.move.x == move.x and child.move.y == move.y:
                return child

        # Create a new child
        new_board = deepcopy(self.board)
        new_board.set_tile_colour(move.x, move.y, self.player)
        child_node = TreeNode(board=new_board, 
                              player=self.player.opposite(), 
                              move=move, 
                              parent=self)
        self.children.append(child_node)
        return child_node
    
    def get_child(self, move):
        """
        Gets the child node for a move.
        Adds the child if it does not exist.
        """
        for child in self.children:
            if child.move.x == move.x and child.move.y == move.y:
                return child
        return self.add_child(move)
    
    @property
    def moves(self) -> set[HeuristicMove]:
        ordered_moves = {
            1: set(),
            2: set(),
            3: set(),
            4: set(),
            5: set()
        }
        for move in self._moves:
            ordered_moves[move.priority].add(move)

        # Only return moves of the highest priority
        for moves in ordered_moves.values():
            if len(moves) > 0:
                return moves
        return set()
    

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
    def find_one_to_connect_moves(board: Board, colour: Colour, chains: set[Chain]) -> set[HeuristicMove]:
        """
        Finds all moves that connect a chain to the edge of the board or to another chain.

        """
        moves = set()

        for chain in chains:
            influence_region = chain.get_influence_region(board)
            for pos in influence_region:
                # Identify game-ending moves (Priority 1)
                if colour == Colour.RED:
                    # Connect edge-connected chain to opposite edge
                    if chain.chain_type == 'Top' and pos[0] == board.size - 1:
                        move = HeuristicMove(pos[0], pos[1], 1)
                        moves.add(move)
                        break
                    elif chain.chain_type == 'Bottom' and pos[0] == 0:
                        move = HeuristicMove(pos[0], pos[1], 1)
                        moves.add(move)
                        break

                    # Connect edge-connected chain to opposite edge-connected chain
                    for other_chain in chains:
                        if chain == other_chain:
                            continue

                        if (chain.chain_type == 'Top' and
                            other_chain.chain_type == 'Bottom' and
                            pos in other_chain.get_influence_region(board)):
                            move = HeuristicMove(pos[0], pos[1], 1)
                            moves.add(move)
                            break

                        if (chain.chain_type == 'Bottom' and
                            other_chain.chain_type == 'Top' and
                            pos in other_chain.get_influence_region(board)):
                            move = HeuristicMove(pos[0], pos[1], 1)
                            moves.add(move)
                            break

                elif colour == Colour.BLUE:
                    # Connect edge-connected chain to opposite edge
                    if chain.chain_type == 'Left' and pos[1] == board.size - 1:
                        move = HeuristicMove(pos[0], pos[1], 1)
                        moves.add(move)
                        break
                    elif chain.chain_type == 'Right' and pos[1] == 0:
                        move = HeuristicMove(pos[0], pos[1], 1)
                        moves.add(move)
                        break

                    # Connect edge-connected chain to opposite edge-connected chain
                    for other_chain in chains:
                        if chain == other_chain:
                            continue

                        if (chain.chain_type == 'Left' and
                            other_chain.chain_type == 'Right' and
                            pos in other_chain.get_influence_region(board)):
                            move = HeuristicMove(pos[0], pos[1], 1)
                            moves.add(move)
                            break
                        if (chain.chain_type == 'Right' and
                            other_chain.chain_type == 'Left' and
                            pos in other_chain.get_influence_region(board)):
                            move = HeuristicMove(pos[0], pos[1], 1)
                            moves.add(move)
                            break
                    

                # Identify priority 2 moves
                if colour == Colour.RED:
                    if 'Top' not in chain.chain_type and pos[0] == 0:
                        move = HeuristicMove(pos[0], pos[1], 2)
                        moves.add(move)
                    if 'Bottom' not in chain.chain_type and pos[0] == board.size - 1:
                        move = HeuristicMove(pos[0], pos[1], 2)
                        moves.add(move)
                elif colour == Colour.BLUE:
                    if 'Left' not in chain.chain_type and pos[1] == 0:
                        move = HeuristicMove(pos[0], pos[1], 2)
                        moves.add(move)
                    if 'Right' not in chain.chain_type and pos[1] == board.size - 1:
                        move = HeuristicMove(pos[0], pos[1], 2)
                        moves.add(move)

                # Check if this move creates a new connection (e.g., joins chains or completes a path)
                connected_chains = [
                    other_chain for other_chain in chains 
                    if other_chain != chain 
                    and pos in other_chain.get_influence_region(board)
                ]
                    
                # Identify priority 3 moves
                for other_chain in connected_chains:
                    if colour == Colour.RED:
                        if (chain.chain_type != 'Top' and chain.chain_type != 'Bottom' and 
                            (other_chain.chain_type == 'Bottom' or other_chain.chain_type == 'Top')):
                            move = HeuristicMove(pos[0], pos[1], 3)
                            moves.add(move)
                            break # Only add the move once
                    elif colour == Colour.BLUE:
                        if (chain.chain_type != 'Left' and chain.chain_type != 'Right' and 
                            (other_chain.chain_type == 'Right' or other_chain.chain_type == 'Left')):
                            move = HeuristicMove(pos[0], pos[1], 3)
                            moves.add(move)
                            break # Only add the move once
                        
        return moves

    @staticmethod
    def find_one_possible_connect_moves(board: Board, colour: Colour, chains: set[Chain]) -> set[HeuristicMove]:
        moves = set()

        for chain in chains:
            influence_region = chain.get_influence_region(board)
            for pos in influence_region:
                neighbour_tiles = Utilities.get_neighbours(board, board.tiles[pos[0]][pos[1]])
                neighbour_positions = [(tile.x, tile.y) for tile in neighbour_tiles] # tiles are unhashable

                # Check for priority 4 moves (one-possible-connect to edge)
                if colour == Colour.RED:
                    # potential future connection to top
                    if 'Top' not in chain.chain_type and 0 in [pos[0] for pos in neighbour_positions]:
                        move = HeuristicMove(pos[0], pos[1], 4)
                        moves.add(move)
                        break # Only add the move once
                    # potential future connection to bottom
                    if 'Bottom' not in chain.chain_type and board.size - 1 in [pos[0] for pos in neighbour_positions]:
                        move = HeuristicMove(pos[0], pos[1], 4)
                        moves.add(move)
                        break # Only add the move once
                elif colour == Colour.BLUE:
                    # potential future connection to left
                    if 'Left' not in chain.chain_type and 0 in [pos[1] for pos in neighbour_positions]:
                        move = HeuristicMove(pos[0], pos[1], 4)
                        moves.add(move)
                        break # Only add the move once
                    # potential future connection to right
                    if 'Right' not in chain.chain_type and board.size - 1 in [pos[1] for pos in neighbour_positions]:
                        move = HeuristicMove(pos[0], pos[1], 4)
                        moves.add(move)
                        break # Only add the move once

                # Check for priority 5 moves (one-possible-connect to another chain which is connected to an edge)

                possible_connected_chains = [
                    other_chain for other_chain in chains 
                    if other_chain != chain 
                    and len(other_chain.get_influence_region(board).intersection(neighbour_positions)) > 0
                ]

                for other_chain in possible_connected_chains:
                    if colour == Colour.RED:
                        if ((chain.chain_type != 'Top' and other_chain.chain_type == 'Top') or 
                            (chain.chain_type != 'Bottom' and other_chain.chain_type == 'Bottom')):
                            move = HeuristicMove(pos[0], pos[1], 5)
                            moves.add(move)
                            break
                    elif colour == Colour.BLUE:
                        if ((chain.chain_type != 'Left' and other_chain.chain_type == 'Left') or 
                            (chain.chain_type != 'Right' and other_chain.chain_type == 'Right')):
                            move = HeuristicMove(pos[0], pos[1], 5)
                            moves.add(move)
                            break
        
        return moves
    
    def find_blockable_moves(board: Board, colour: Colour, chains: set[Chain]) -> set[HeuristicMove]:
        """
        Finds one-to-connect moves with only one possible connection to the edge or another chain.
        These moves are blockable by the opposing colour.
        """
        moves = set()

        pairs = itertools.combinations(chains, 2)

        for chain1, chain2 in pairs:
            influence_region_1 = chain1.get_influence_region(board)
            influence_region_2 = chain2.get_influence_region(board)
            connections = influence_region_1 & influence_region_2

            # Exactly one connection is a blockable move
            if len(connections) == 1:
                pos = connections.pop()

                if colour == Colour.RED:
                    type1, type2 = 'Top', 'Bottom'
                elif colour == Colour.BLUE:
                    type1, type2 = 'Left', 'Right'

                if ((chain1.chain_type == type1 and chain2.chain_type == type2) or
                    (chain1.chain_type == type2 and chain2.chain_type == type1)):
                    moves.add(HeuristicMove(pos[0], pos[1], 1))
                elif ((chain1.chain_type == type1 and chain2.chain_type != type2) or
                      (chain1.chain_type == type2 and chain2.chain_type != type1)):
                    moves.add(HeuristicMove(pos[0], pos[1], 2))
                elif ((chain1.chain_type != type1 and chain2.chain_type == type2) or
                      (chain1.chain_type != type2 and chain2.chain_type == type1)):
                    moves.add(HeuristicMove(pos[0], pos[1], 2))
                else: # Both chains are not connected to an edge
                    moves.add(HeuristicMove(pos[0], pos[1], 4))

        return moves
