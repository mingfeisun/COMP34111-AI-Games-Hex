import pytest

from src.Board import Board
from src.Colour import Colour


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
    count_0 = 0
    for i, row in enumerate(board_array):
        for j, tile in enumerate(row):
            if tile == "0":
                count_0 += 1
    b = Board(board_size=len(board_array))
    for i, row in enumerate(board_array):
        for j, tile in enumerate(row):
            if tile == "R":
                b.set_tile_colour(i, j, Colour.RED)
            elif tile == "B":
                b.set_tile_colour(i, j, Colour.BLUE)
    return b