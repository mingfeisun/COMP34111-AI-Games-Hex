import argparse
import importlib
import sys

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hex",
        description="Run a game of Hex. By default, two naive agents will play.",
    )
    parser.add_argument(
        "-p1",
        "--player1",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 1 agent, format: agents.GroupX.AgentFile AgentClassName .e.g. agents.Group0.NaiveAgent NaiveAgent",
    )
    parser.add_argument(
        "-p1Name",
        "--player1Name",
        default="Red",
        type=str,
        help="Specify the player 1 name",
    )
    parser.add_argument(
        "-p2",
        "--player2",
        default="agents.DefaultAgents.NaiveAgent NaiveAgent",
        type=str,
        help="Specify the player 2 agent, format: agents.GroupX.AgentFile AgentClassName .e.g. agents.Group0.NaiveAgent NaiveAgent",
    )
    parser.add_argument(
        "-p2Name",
        "--player2Name",
        default="Blue",
        type=str,
        help="Specify the player 2 name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-b",
        "--board_size",
        type=int,
        default=11,
        help="Specify the board size",
    )
    parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        type=str,
        default=sys.stderr,
        const="game.log",
        help=(
            "Save moves history to a log file,"
            "if the flag is present, the result will be saved to game.log."
            "If a filename is provided, the result will be saved to the provided file."
            "If the flag is not present, the result will be printed to the console, via stderr."
        ),
    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)
    g = Game(
        player1=Player(
            name=args.player1Name,
            agent=getattr(p1, p1_class)(Colour.RED),
        ),
        player2=Player(
            name=args.player2Name,
            agent=getattr(p2, p2_class)(Colour.BLUE),
        ),
        board_size=args.board_size,
        logDest=args.log,
        verbose=args.verbose,
    )
    g.run()
