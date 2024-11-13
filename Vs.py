"""
Program that runs two agents against each other in n games of Hex
Once done it will print the results of the games
"""
import argparse
import importlib
import concurrent.futures

from src.Colour import Colour
from src.Game_No_Logging import Game
from src.Player import Player

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VS Program",
        description="Runs n games of Hex between two agents and prints the results",
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
        default="Player 1",
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
        default="Player 2",
        type=str,
        help="Specify the player 2 name",
    )
    parser.add_argument(
        "-n",
        "--num_games",
        type=int,
        default=10,
        help="Specify the number of games to play",
    )

    args = parser.parse_args()
    p1_path, p1_class = args.player1.split(" ")
    p2_path, p2_class = args.player2.split(" ")
    p1 = importlib.import_module(p1_path)
    p2 = importlib.import_module(p2_path)

    def run_game(game):
        result = game.run()
        print("Game finished")
        return result["winner"]

    games = [Game(
        player1=Player(
            name=args.player1Name,
            agent=getattr(p1, p1_class)(Colour.RED),
        ),
        player2=Player(
            name=args.player2Name,
            agent=getattr(p2, p2_class)(Colour.BLUE),
        ),
    ) for _ in range(args.num_games)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_game, games))

    print("Results:")
    print(f"{args.player1Name} wins:", results.count(args.player1Name))
    print(f"{args.player2Name} wins:", results.count(args.player2Name))