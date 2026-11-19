import subprocess

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.agent_process = subprocess.Popen(
            ["./agents/MCTSAgent/mcts-hex"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            # stderr is not read by the engine; a PIPE here would fill up and
            # block the agent, so its debug output goes to the console instead
            stderr=None,
            text=True,
            bufsize=1
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

        # protocol format: rows joined by commas, R/B/0 per tile
        board_string = board.to_protocol_string()

        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.x == -1 and opp_move.y == -1:
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().rstrip()
        if not response:
            raise RuntimeError(
                "External agent sent no move (exit code "
                f"{self.agent_process.poll()}): it may have crashed, or "
                "written its move without a newline or without flushing stdout."
            )
        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        x, y = map(int, response.split(","))
        return Move(x, y)
