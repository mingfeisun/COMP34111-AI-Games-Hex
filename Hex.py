import argparse
import importlib
import random
import sys

from src.AgentProxy import AgentLoadError, AgentProxy, pausing_supported
from src.Colour import Colour
from src.Game import Game
from src.Player import Player

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Hex",
        description=(
            "Run a game of Hex. By default, two naive agents will play. Each agent "
            "runs in its own process and is paused during the opponent's turn, as "
            "in marked games."
        ),
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

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for this game (player 1 gets SEED, player 2 SEED+1). "
            "Without it, the engine chooses one and prints it, so that you can "
            "repeat a game."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: run both agents inside the engine's process, "
            "so that you can use a debugger (pdb, breakpoint()) in "
            "your agent. Agents are not paused, and there are no time limits; the "
            "engine prints the total time of each player at the end. Marked games "
            "do not use this."
        ),
    )

    args = parser.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    print(f"Seed: {seed}", file=sys.stderr)

    in_process = args.debug
    if not in_process and not pausing_supported():
        print(
            "This system (Windows?) cannot pause processes: running in debug mode "
            "(--debug). Use the Docker container to test as in marked games.",
            file=sys.stderr,
        )
        in_process = True

    specs = [
        (args.player1, args.player1Name, Colour.RED, seed),
        (args.player2, args.player2Name, Colour.BLUE, seed + 1),
    ]
    players = []
    for i, (spec, name, colour, agent_seed) in enumerate(specs):
        if in_process:
            path, cls = spec.split(" ")
            random.seed(agent_seed)
            agent = getattr(importlib.import_module(path), cls)(colour)
        else:
            # started one after the other, so that each start-up gets all CPUs;
            # each agent is paused once its start-up has finished
            try:
                agent = AgentProxy(spec, colour, args.board_size, seed=agent_seed)
            except AgentLoadError as e:
                for p in players:
                    p.agent.close()
                other = specs[1 - i][1]
                Game.write_failed_load(args.log, loser=name, winner=other, reason=str(e))
                sys.exit(0)
            print(f"{name}: start-up {agent.startup_s:.2f} s", file=sys.stderr)
        players.append(Player(name=name, agent=agent))

    g = Game(
        player1=players[0],
        player2=players[1],
        board_size=args.board_size,
        logDest=args.log,
        verbose=args.verbose,
        time_limit=not args.debug,
    )
    result = g.run()

    # total move time of each player (start-up not included)
    if result:
        limit_s = Game.MAXIMUM_TIME / 10**9
        print("Total move time (start-up not included):", file=sys.stderr)
        for key in ("player1", "player2"):
            used_s = result[f"{key}_move_time"]
            note = ""
            if used_s > limit_s:
                note = (
                    f"  -- more than the limit of {limit_s:.0f} s: in a marked game "
                    "this player loses by TIMEOUT"
                )
            print(f"  {result[key]}: {used_s:.3f} s{note}", file=sys.stderr)
        if args.debug:
            print(
                f"  (debug mode: the {limit_s:.0f}-second limit is not applied, and "
                "time spent in a debugger counts)",
                file=sys.stderr,
            )
