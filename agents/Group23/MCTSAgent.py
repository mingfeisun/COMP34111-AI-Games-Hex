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
        mcts = MCTS(self.colour, max_simulation_length=self.max_simulation_length)
        best_move, _ = mcts.run(board)

        return best_move