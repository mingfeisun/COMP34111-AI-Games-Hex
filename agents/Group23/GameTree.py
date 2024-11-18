from src.Board import Board
from src.Colour import Colour
from src.Move import Move

class GameTree:
    def __init__(self, board: Board, colour: Colour, move: Move = None):
        self.board = board
        self.colour = colour
        self.children = []
        self.move = move # represents the move that led to this state

        self.num_visits = 0
        self.value = 0

    def add_move(self, move):
        new_colour = Colour.opposite(self.colour)
        new_state = GameTree(self.board.move(move), new_colour, move)
        self.children.append(new_state, new_colour)
        return new_state

    def get_node(self, board):
        for child in self.children:
            if child.board == board:
                return child
        return None