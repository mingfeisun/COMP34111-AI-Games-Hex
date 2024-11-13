import pytest

from src.Board import Board
from src.Colour import Colour

def board_from_array(board_array: list[str]) -> Board:
    b = Board(board_size=len(board_array))
    for i, row in enumerate(board_array):
        for j, tile in enumerate(row):
            if tile == "R":
                b.set_tile_colour(i, j, Colour.RED)
            elif tile == "B":
                b.set_tile_colour(i, j, Colour.BLUE)
    return b


@pytest.fixture(scope="function")
def blue_win_board():
    """
    Fixture for a board where blue wins
    R R R R B B R B R R R
     R B R B R B R B R B B
      R B R R R R R B B B B
       R R R R 0 R B 0 R B R
        R R B B R B B B R 0 B
         R R B R B 0 R 0 0 0 0
          0 R 0 R B B B B B 0 B
           B B 0 R 0 B R 0 0 R 0
            0 0 B R R B 0 B B B 0
             B B B R B R 0 R B B 0
              0 R B B R 0 0 B 0 R 0
    """
    board_array = [
        "RRRRBBRBRRR",
        "RBRBRBRBRBB",
        "RBRRRRRBBBB",
        "RRRR0RB0RBR",
        "RRBBRBBBR0B",
        "RRBRB0R0000",
        "0R0RBBBBB0B",
        "BB0R0BR00R0",
        "00BRRB0BBB0",
        "BBBRBR0RBB0",
        "0RBBR00B0R0",
    ]
    return board_from_array(board_array)

@pytest.fixture(scope="function")
def one_away_board():
    """
    Fixture for a board where blue is one move away from winning
    R R R R B B R B R R R
     R B R B R B R B 0 0 0
      R B R R R 0 R B B 0 0
       R R R 0 0 R B 0 0 0 0
        R R B B R B B B 0 0 0
         R R B R B 0 R 0 0 0 0
          0 R 0 R B B B B B 0 B
           B B 0 R 0 B 0 0 0 R 0
            0 0 B R R B 0 B B 0 0
             B B B R B R 0 0 B 0 0
              0 R B B R 0 0 B 0 R 0
    """

    array_board = [
        "RRRRBBRBRRR",
        "RBRBRBRB000",
        "RBRRR0RBB00",
        "RRR00RB0000",
        "RRBBRBBB000",
        "RRBRB0R0000",
        "0R0RBBBBB0B",
        "BB0R0B000R0",
        "00BRRB0BB00",
        "BBBRBR0B000",
        "0RBBR00B0R0",
    ]
    return board_from_array(array_board)
