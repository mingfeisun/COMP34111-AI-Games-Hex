from subprocess import PIPE, Popen

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class ExternalAgent(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        # spawn a process that calls a compiled java NaiveAgent.class file and passes two arguments:
        # - "R" or "B" to tell the agent which colour it is
        # - 11, which is the size of the board
        self.agent_process = Popen(
            ["java", "-cp", "agents/DefaultAgents", "NaiveAgent", colour.get_char(), "11"],
            stdout=PIPE,
            stdin=PIPE,
            text=True,
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
        # translate the python objects into string representations
        rows = board.tiles()
        board_strings = []
        for row in rows:
            row_string = ""
            for tile in row:
                colour = tile.colour
                if colour is None:
                    t = "0"
                else:
                    t = colour.get_char()
                row_string += t
            board_strings.append(row_string)
        board_string = ",".join(board_strings)

        if opp_move is None:
            # should be turn 1
            command = f"START;;{board_string};{turn};"
        elif opp_move.is_swap():
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        # send the command to the agent process and get response
        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().rstrip()

        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        x, y = response.split(",")
        return Move(int(x), int(y))
