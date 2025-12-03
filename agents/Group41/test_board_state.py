import sys
import os

# Add project root to import path so "src" becomes visible
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


from src.Board import Board
from src.Move import Move
from src.Colour import Colour
from agents.Group41.board_state import BoardStateNP

def test_numpy_conversion():
    b = Board(3)
    b.set_tile_colour(0, 1, Colour.RED)
    b.set_tile_colour(2, 2, Colour.BLUE)

    w = BoardStateNP(b)
    arr = w.get_numpy()

    print(arr)
    # Expected:
    # [[0 1 0]
    #  [0 0 0]
    #  [0 0 2]]
def test_is_legal():
    b = Board(3)
    b.set_tile_colour(1, 1, Colour.RED)

    w = BoardStateNP(b)
    assert w.is_legal(0, 0) == True
    assert w.is_legal(1, 1) == False  # occupied
    assert w.is_legal(-1, 0) == False # out of bounds
    assert w.is_legal(3, 2) == False # out of bounds

    print("is_legal tests passed!")

def test_apply_move():
    b = Board(3)
    w = BoardStateNP(b)

    m = Move(1, 2)
    w.apply_move(m, Colour.BLUE)

    arr = w.get_numpy()
    print(arr)

    assert arr[1, 2] == 2   # blue
    assert b.tiles[1][2].colour == None  # engine board should NOT change

    print("apply_move tests passed!")

def test_clone():
    b = Board(3)
    w1 = BoardStateNP(b)
    w2 = w1.clone()

    # Modify clone
    w2.apply_move(Move(0, 0), Colour.RED)

    # Original should remain unchanged
    assert w1.get_numpy()[0, 0] == 0
    assert w2.get_numpy()[0, 0] == 1

    print("clone tests passed!")

def test_get_numpy():
    b = Board(3)
    w = BoardStateNP(b)

    arr1 = w.get_numpy()
    arr1[0, 0] = 9

    arr2 = w.get_numpy()
    assert arr2[0, 0] == 9  # they reference the same object

    print("get_numpy tests passed!")


test_numpy_conversion()
test_is_legal()
test_apply_move()
test_clone()
test_get_numpy()