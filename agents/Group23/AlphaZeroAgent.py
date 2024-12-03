from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from agents.TestAgents.utils import make_valid_move

class AlphaZeroAgent(AgentBase):

    _board_size: int = 11
    _board_history = []
    _trained_policy_value_network = None


    def __init__(self, colour: Colour, custom_trained_network=None):
        super().__init__(colour)
        self._trained_policy_value_network = custom_trained_network

    def get_input_vector(self, board: Board) -> list[int]:
        """generate input vector for neural network
        based on current and recent board states

        Args:
            board (Board): current board state

        Returns:
            list[int]: input vector for neural network
        """
        # maintain history of last 5 board states
        if len(self._board_history) >= 5:
            self._board_history.pop(0)
        
        # convert board state to input vector
        input_vector = []
        for i in range(self._board_size):
            new_line = []
            for j in range(self._board_size):
                tile = board.tiles[i][j].colour
                if tile == None:
                    new_line.append(0)
                elif tile == self.colour:
                    new_line.append(1)
                else:
                    new_line.append(-1)
            input_vector.append(new_line)

        self._board_history.append(input_vector)

        # flatten history into a 1D list
        flattened_history = []
        for line in self._board_history:
            for item in line:
                for value in item:
                    flattened_history.append(value)
        
        return flattened_history

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        input_vector = self.get_input_vector(board)
        # print(input_vector)

        if turn == 2:
            return Move(-1, -1)
        else:
            return make_valid_move(board)
