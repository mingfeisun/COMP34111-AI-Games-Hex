from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.TestAgents.CheatMove import CheatMove

class CrashAgent(AgentBase):

    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        return CheatMove(10,13)
