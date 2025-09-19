import argparse
import csv
import importlib
import logging
import re
import traceback
from datetime import datetime
from glob import glob
from itertools import permutations, repeat
from multiprocessing import Pool, TimeoutError

from src.Colour import Colour
from src.EndState import EndState
from src.Game import Game, format_result
from src.Player import Player

# Set up logger
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# set the timeout limit in seconds. Set to 6 minutes which i twice the amount of time assigned to each agent (3 minuets)
TIME_OUT_LIMIT = 6 * 60

fieldnames = [
    "player1",
    "player2",
    "winner",
    "win_method",
    "player1_move_time",
    "player2_move_time",
    "player1_turns",
    "player2_turns",
    "total_turns",
    "total_game_time",
]
time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def get_player_name(agentPair: tuple[str, str]):
    """Extract player names from the agent pair."""
    player1, player2 = agentPair
    p1Name = player1.split(".")[1] if "." in player1 else player1
    p2Name = player2.split(".")[1] if "." in player2 else player2
    return p1Name, p2Name


def get_results_for_game_global_timeout(player1, player2, logDest, errorGameListPath):
    def _get_winner(player1, player2):
        """Determine the winner based on log files."""
        winner_name = ""
        if os.path.exists(logDest):
            with open(logDest, "r") as logFile:
                lines = logFile.readlines()
                if "winner" in lines[-1].lower():
                    winner_name = lines[-1].split(",")[1].strip()
                else:
                    last_mover = lines[-1].strip()
                    if last_mover == player1:
                        winner_name = player1
                    elif last_mover == player2:
                        winner_name = player2
                    else:
                        winner_name = "Unknown"
        else:
            with open(errorGameListPath, "r") as errFile:
                lines = errFile.readlines()
                for l in lines:
                    p1, p2, errpr = l.split(",")
                    if player1 in p1 and player2 in p2:
                        winner_name = player2

        return winner_name

    player1 = player1.split(" ")[0].split(".")[1]
    player2 = player2.split(" ")[0].split(".")[1]
    winner = _get_winner(player1, player2)
    return format_result(
        player1_name=player1,
        player2_name=player2,
        winner=winner,
        win_method=EndState.TIMEOUT,
        player_1_move_time=0,
        player_2_move_time=0,
        player_1_turn=0,
        player_2_turn=0,
        total_turns=0,
        total_time=0,
    )


def run(games: list[tuple[str, str]]):
    """Run the tournament. This uses multiprocessing to dispatch each of the game.
    The results are written to a csv file.
    The error will be written to a log file.

    Args:
        games (list[tuple[str, str]]): all the games pair that need to be played
    """
    resultFilePath = f"game_results_{time}.csv"
    errorGameListPath = f"error_game_list_{time}.log"

    with open(resultFilePath, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()

    # Run the tournament
    gameResults = []
    with Pool() as pool:
        # using apply_async to apply the game function to the player pair
        p1Name, p2Name = get_player_name(games[0])
        logDST = f"./all_game_logs_{time}/{p1Name}_vs_{p2Name}.log"
        result = []
        for agentPair in games:
            p1Name, p2Name = get_player_name(agentPair)
            logDST = f"./all_game_logs_{time}/{p1Name}_vs_{p2Name}.log"
            result.append(
                pool.apply_async(
                    run_match,
                    (agentPair, logDST),
                )
            )

        # gather all the results. Error of a game is captured and written to a log file.
        for i, gameResult in enumerate(result):
            game_finished = False
            try:
                r = gameResult.get(timeout=TIME_OUT_LIMIT)
                with open(resultFilePath, "a", newline="") as csvFile:
                    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                    writer.writerow(r)
                gameResults.append(r)
                game_finished = True

            except TimeoutError as error:
                logger.warning(f"Timed out between {games[i]}")
                with open(errorGameListPath, "a") as errFile:
                    errFile.write(f"{games[i]}, {repr(error)}\n")

            except Exception as error:
                logger.error(f"Exception occurred between {games[i]}: {repr(error)}")
                logger.error(traceback.format_exc())
                with open(errorGameListPath, "a") as errFile:
                    errFile.write(
                        f"{games[i]}, {repr(error), {traceback.format_exc()}}\n"
                    )
            finally:
                if not game_finished:
                    # if the game did not finish, we need to write the result to the csv file
                    r = get_results_for_game_global_timeout(
                        games[i][0],
                        games[i][1],
                        logDest=logDST,
                        errorGameListPath=errorGameListPath,
                    )

                    with open(resultFilePath, "a", newline="") as csvFile:
                        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                        writer.writerow(r)
                    gameResults.append(r)

    export_stats(gameResults)


def run_match(agentPair: tuple[str, str], logDest: str) -> dict:
    """Run a single game between two agents. It parses the agent string pair
    and creates the player objects.
    """

    player1, player2 = agentPair
    logger.info(f"starting game between {player1} and {player2}")

    player1_class = None
    player2_class = None

    try:
        p1_path, p1_class = player1.split(" ")
        p1 = importlib.import_module(p1_path)
        # this should return the group name
        p1Name = player1.split(".")[1]
        player1_class = Player(
            name=p1Name,
            agent=getattr(p1, p1_class)(Colour.RED),
        )
    except ModuleNotFoundError as error:
        logger.error(
            f"Exception occured importing {player1}, agent file could not be imported: {repr(error)}"
        )
        logger.error(traceback.format_exc())
    except Exception as error:
        logger.error(f"Exception occured importing {player1}: {repr(error)}")
        logger.error(traceback.format_exc())

    try:
        p2_path, p2_class = player2.split(" ")
        p2 = importlib.import_module(p2_path)
        # this should return the group name
        p2Name = player2.split(".")[1]
        player2_class = Player(
            name=p2Name,
            agent=getattr(p2, p2_class)(Colour.BLUE),
        )
    except ModuleNotFoundError as error:
        logger.error(
            f"Exception occured importing {player1}, agent file could not be imported: {repr(error)}"
        )
        logger.error(traceback.format_exc())
    except Exception as error:
        logger.error(f"Exception occured importing {player1}: {repr(error)}")
        logger.error(traceback.format_exc())

    if player1_class is None and player2_class is None:
        logger.info(
            f"Both agents failed to load for game between {player1} and {player2}."
        )
        result = format_result(
            player1_name=player1.split(".")[1],
            player2_name=player2.split(".")[1],
            winner="",
            win_method=EndState.FAILED_LOAD,
            player_1_move_time="",
            player_2_move_time="",
            player_1_turn="",
            player_2_turn="",
            total_turns="",
            total_time="",
        )
    elif player1_class is None:
        logger.info(f"Agent {player1} failed to load, {player2} wins.")
        result = format_result(
            player1_name=player1.split(".")[1],
            player2_name=player2_class.name,
            winner=player2_class.name,
            win_method=EndState.FAILED_LOAD,
            player_1_move_time="",
            player_2_move_time="",
            player_1_turn="",
            player_2_turn="",
            total_turns="",
            total_time="",
        )
    elif player2_class is None:
        logger.info(f"Agent {player2} failed to load, {player1} wins.")
        result = format_result(
            player1_name=player1_class.name,
            player2_name=player2.split(".")[1],
            winner=player1_class.name,
            win_method=EndState.FAILED_LOAD,
            player_1_move_time="",
            player_2_move_time="",
            player_1_turn="",
            player_2_turn="",
            total_turns="",
            total_time="",
        )
    else:
        # the getattr is to get the class from the module, then instantiate it with the colour
        g = Game(
            player1=player1_class,
            player2=player2_class,
            board_size=11,
            silent=True,
            logDest=logDest,
        )
        result = g.run()
        logger.info(f"Game complete normally between {player1} and {player2}")

    return result


def export_stats(gameResults: list[dict]):
    playerStats = {}

    statEntry = {
        "matches": 0,
        "wins": 0,
        "win_rate": 0,
        "total_move_time": 0,
        "total_moves": 0,
        "average_move_time": 0,
        "illegal_moves_loss": 0,
        "time_out_loss": 0,
        "regular_loss": 0,
    }

    # populate the player stats dictionary
    for result in gameResults:
        for player in [result["player1"], result["player2"]]:
            if player not in playerStats:
                playerStats[player] = statEntry.copy()

    # fill in the data
    for result in gameResults:
        player1 = result["player1"]
        player2 = result["player2"]
        winner = result["winner"]
        loser = player2 if player1 == winner else player1
        if result["win_method"] == EndState.FAILED_LOAD:
            if winner == "":
                continue
            else:
                playerStats[winner]["matches"] += 1
                playerStats[winner]["wins"] += 1
                playerStats[loser]["matches"] += 1
        else:
            if winner == player1:
                loser = player2
            else:
                loser = player1

            playerStats[player1]["matches"] += 1
            playerStats[player2]["matches"] += 1
            playerStats[player1]["total_move_time"] += result["player1_move_time"]
            playerStats[player2]["total_move_time"] += result["player2_move_time"]
            playerStats[player1]["total_moves"] += result["player1_turns"]
            playerStats[player2]["total_moves"] += result["player2_turns"]

            playerStats[winner]["wins"] += 1
            playerStats[loser]["illegal_moves_loss"] += (
                1 if result["win_method"] == "BAD_MOVE" else 0
            )
            playerStats[loser]["time_out_loss"] += (
                1 if result["win_method"] == "TIMEOUT" else 0
            )
            playerStats[loser]["regular_loss"] += (
                1 if result["win_method"] == "WIN" else 0
            )

    for player, stats in playerStats.items():
        playerStats[player]["win_rate"] = (
            stats["wins"] / stats["matches"] if stats["matches"] > 0 else 0
        )
        playerStats[player]["average_move_time"] = (
            stats["total_move_time"] / stats["total_moves"]
            if stats["total_moves"] > 0
            else 0
        )

    with open(f"game_stat_{time}.csv", "w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["player"] + list(statEntry.keys()))
        for player, stats in playerStats.items():
            writer.writerow([player] + list(stats.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Hex Tournament. Will create the game stat file and all the game log. In the event of crashing, the error event will go into the error log"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-p",
        "--partialTournament",
        type=str,
        help="Path to a newline separated list of int, which are the group number. Each line will play against every other group",
    )

    args = parser.parse_args()

    def extract_group_number(path):
        if match := re.search(r"Group(\d+)", path):
            return int(match.group(1))
        else:
            raise ValueError(f"Invalid path {path}")

    agents = {}
    for p in sorted(glob("agents/Group*/cmd.txt"), key=extract_group_number):
        with open(p, "r") as f:
            agent = f.read().split("\n")[0].strip()
            if p.split("/")[1] != agent.split(".")[1]:
                print(
                    f"Agent location {agent} does not match group number for path {p}, agent will not be loaded."
                )
            else:
                agents[extract_group_number(p)] = agent

    games = []
    games = list(permutations(agents.values(), 2))

    # if running a partial tournament, the following will overwrite the games list
    # Each item on the partial list will play against every agent in the agents directory
    if args.partialTournament:
        with open(args.partialTournament) as f:
            for line in f:
                # the given line as player A
                games.extend(
                    zip(repeat(agents[int(line)]), list(agents.values()), strict=False)
                )
                # the given line as player B
                games.extend(
                    zip(agents.values(), repeat(agents[int(line)]), strict=False)
                )

        # remove all repeat and self play
        games = [(i, j) for i, j in list(set(games)) if i != j]

    run(games)
