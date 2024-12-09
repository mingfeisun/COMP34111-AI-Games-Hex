from random import choice

import sys
import time
import os
sys.path.append(r'/home/hex/agents/Group25-AZ')
sys.path.append(r'/home/hex/agents/Group25-AZ/azsrc')
sys.path.append(r'/home/hex/agents/Group25-AZ/azsrc/hexhex')
sys.path.append(r'/home/hex/agents/Group25-AZ/azsrc/model')

# print files in a given directory
print(os.listdir(r'/home/hex/agents/Group25-AZ/azsrc/model'))


from azsrc.hexhex.utils.utils import load_model
from azsrc.hexhex.logic import hexboard
from azsrc.hexhex.logic.hexgame import MultiHexGame

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

class AZAgent(AgentBase):
    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        self.model = load_model(r"/home/hex/agents/Group25-AZ/azsrc/data/azagnet-1000.pt")
        self.hexhexboard = hexboard.Board(11, False)
        self.game = MultiHexGame(
            boards=(self.hexhexboard,),
            models=(self.model,),
            noise=None,
            noise_parameters=None,
            temperature=0.0,
            temperature_decay=1.0,
        )
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """
        if opp_move is not None and opp_move.x == -1 and opp_move.y == -1:
            self.hexhexboard.switch = True
        # update internal hexhexboard
        if opp_move is not None and opp_move.x != -1:
            self.hexhexboard.set_stone((opp_move.x, opp_move.y))

        self.game.batched_single_move(self.model)
        move = self.hexhexboard.move_history[-1][1]
        alpha, numeric = hexboard.position_to_alpha_numeric(move)
        print(f'moving to {move}')
        print(f'ai move: {alpha}{numeric}') # debug
        print(opp_move)
        print(self.hexhexboard.move_history)
        # time.sleep(5)
        return Move(int(numeric) - 1, ord(alpha) - ord('a'))
