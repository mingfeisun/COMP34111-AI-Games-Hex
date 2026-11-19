from agents.TestAgents.CheatMove import CheatMove
from agents.TestAgents.utils import make_valid_move
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class CheatAgent(AgentBase):
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn < 10:
            return make_valid_move(board)
        else:
            return self.make_cheat_move(board)

    def make_cheat_move(self, board: Board) -> Move:
        for i in range(board.size):
            for j in range(board.size):
                t = board.tiles[i][j]
                if t.colour is not self.colour:
                    return CheatMove(i, j)
