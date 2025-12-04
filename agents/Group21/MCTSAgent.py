from agents.Group21.MCTS import MCTS
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.mcts = MCTS(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # Update MCTS with current turn number, opponents move, select a move using MCTS and then update again
        self.mcts.current_turn = turn
        self.mcts.update(board, opp_move)
        move = self.mcts.run()
        self.mcts.update(board, move)
        return move

