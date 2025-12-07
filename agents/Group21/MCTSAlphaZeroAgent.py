from agents.Group21.AlphaZeroMCTS import AlphaZeroMCTS
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import numpy as np

class MCTSAlphaZeroAgent(AgentBase):
    def __init__(self, colour: Colour, neural_net, simulations=50):
        self.colour = colour
        self.mcts = AlphaZeroMCTS(neural_net)
        self.simulations = simulations

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        legal_moves, pi = self.mcts.run(board, simulations=self.simulations)
        # Select move_index with highest probability in pi distribution
        move_index = np.argmax(pi)
        return legal_moves[move_index]

