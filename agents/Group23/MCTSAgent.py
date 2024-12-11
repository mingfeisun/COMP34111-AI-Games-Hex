import logging
from random import choice

from agents.Group23.treenode import TreeNode
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

from agents.Group23.mcts import MCTS
from agents.Group23.zobrist_hasher import ZobristHasher

class MCTSAgent(AgentBase):
    """An agent that uses MCTS for Hex."""
    logger = logging.getLogger(__name__)

    # Strong opening moves for the agent.
    # Red moves first, then Blue.
    # Based on https://www.hexwiki.net/index.php/Openings_on_11_x_11#Without_swap
    strong_opening_moves = [
        Move(1, 9), Move(1,10),
        Move(2, 1), Move(2, 2), Move(2, 3), Move(2, 4), Move(2, 6), Move(2, 7), Move(2, 8), Move(2, 9),
        Move(3, 1), Move(3, 2), Move(3, 3), Move(3, 4), Move(3, 5), Move(3, 6), Move(3, 7), Move(3, 8), Move(3, 9),
        Move(4, 0), Move(4, 1), Move(4, 2), Move(4, 3), Move(4, 4), Move(4, 5), Move(4, 6), Move(4, 7), Move(4, 8), Move(4, 9),
        Move(5, 1), Move(5, 2), Move(5, 3), Move(5, 4), Move(5, 5), Move(5, 6), Move(5, 7), Move(5, 8), Move(5, 9),
        Move(6, 1), Move(6, 2), Move(6, 3), Move(6, 4), Move(6, 5), Move(6, 6), Move(6, 7), Move(6, 8), Move(6, 9), Move(6, 10),
        Move(7, 1), Move(7, 2), Move(7, 3), Move(7, 4), Move(7, 5), Move(7, 6), Move(7, 7), Move(7, 8), Move(7, 9),
        Move(8, 1), Move(8, 2), Move(8, 3), Move(8, 4), Move(8, 6), Move(8, 7), Move(8, 8), Move(8, 9),
        Move(9, 0), Move(9, 1)
    ]

    fair_opening_moves = [
        Move(0, 10),
        Move(1, 2), Move(0, 8),
        Move(2, 0), Move(2, 5), Move(2, 10),
        Move(3, 0), Move(3, 10),
        Move(4, 10),
        Move(5, 0), Move(5, 10),
        Move(6, 0),
        Move(7, 0), Move(7, 10),
        Move(8, 0), Move(8, 5), Move(8, 10),
        Move(9, 2), Move(9, 8),
        Move(10, 0)
    ]

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5):
        super().__init__(colour)
        self.max_simulation_length = max_simulation_length # max length of a simulation
        self.root = None
        self.zobrist_hasher = ZobristHasher(11)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Selects a move using MCTS."""
        # First move - choose a fair move
        if opp_move == None:
            self.logger.info('First move, choosing a fair move.')
            return choice(self.fair_opening_moves)

        if turn == 2 and opp_move in self.strong_opening_moves:
            # If the opponent makes a strong opening move, use the pie rule to swap.
            self.logger.info('Opponent made a strong opening move, swapping.')
            return Move(-1, -1)
        
        # Look up move in transpotition table (if available)
        hash = self.zobrist_hasher.hash(board)
        move = self.zobrist_hasher.get_move(hash)
        if move is not None:
            self.logger.info(f'!!!!!!!!!!!!!!!!!!!!!!!!!! Found a move in the transposition table: ({move[0]}, {move[1]}) !!!!!!!!!!!!!!!!!!!!!!!!!!')
            return Move(move[0], move[1])

        turn_length = self.allowed_time(turn)
        mcts = MCTS(self.colour, max_simulation_length=turn_length)

        if self.root is None:
            self.root = TreeNode(board=board, player=self.colour)

        self.root = self.root.get_child(opp_move) # Update the node to the child corresponding to the opponent's move
        self.root, _ = mcts.run(self.root)

        print(f'====================')
        print(f'Chose best child:')
        print(f' - Move: {self.root.move}')
        print(f' - Wins: {self.root.wins}')
        print(f' - Visits: {self.root.visits}')
        print(f'From {len(self.root.children)} possible moves.')
        print(f'====================')

        return self.root.move
    
    def allowed_time(self, turn_number, total_turns=121, total_time=300):
        """
        Calculate the allowed time for a turn in a game, giving more time to earlier turns
        and less to later ones.

        Args:
            turn_number (int): The current turn number (1-indexed).
            total_turns (int): Total number of turns in the game (default: 121).
            total_time (float): Total allowed time in seconds (default: 300 seconds).

        Returns:
            float: Allowed time in seconds for the given turn.
        """
        # Parameter controlling the decay rate (higher values give more time to early turns).
        decay_rate = 0.98

        # Calculate the weight for each turn based on the decay rate.
        weights = [decay_rate ** i for i in range(total_turns)]
        total_weight = sum(weights)

        # Calculate the allowed time for the current turn.
        turn_weight = weights[turn_number - 1]  # Turn number is 1-indexed.
        turn_time = (turn_weight / total_weight) * total_time

        print(f'Allocated {turn_time:.2f}s for turn {turn_number}')
        return turn_time