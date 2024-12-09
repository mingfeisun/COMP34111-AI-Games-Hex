import logging

from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour

from agents.Group23.mcts import MCTS

class MCTSAgent(AgentBase):
    """An agent that uses MCTS for Hex."""
    logger = logging.getLogger(__name__)

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5):
        super().__init__(colour)
        self.max_simulation_length = max_simulation_length # max length of a simulation

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Selects a move using MCTS."""
        turn_length = self.allowed_time(turn)
        mcts = MCTS(self.colour, max_simulation_length=turn_length)
        best_move, _ = mcts.run(board)

        return best_move
    
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