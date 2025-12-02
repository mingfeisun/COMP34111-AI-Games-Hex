from random import choice, random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class RandomAgent(AgentBase):
    """
    A smarter naive agent that checks the board before returning a move, preventing illegal moves.
    """

    _board_size: int = 11
    _choices: list[tuple[int, int]]

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # Swap with 50% chance
        if turn == 2 and choice([True, False]):
            return Move(-1, -1)
        # Otherwise get the list of valid moves by scanning through the board and return a random one
        else:
            valid_moves = [
                (i, j)
                for i in range(board.size)
                for j in range(board.size)
                if not board.tiles[i][j].colour
            ]

            x, y = choice(valid_moves)
            return Move(x, y)