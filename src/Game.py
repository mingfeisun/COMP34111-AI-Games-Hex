import copy
import logging
import os
import sys
from time import perf_counter_ns as time
from typing import TextIO

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.EndState import EndState
from src.Move import Move
from src.Player import Player

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(levelname)s]-%(asctime)s - %(message)s", level=logging.INFO
)


def format_result(
    *,
    player1_name,
    player2_name,
    winner,
    win_method,
    player_1_move_time,
    player_2_move_time,
    player_1_turn,
    player_2_turn,
    total_turns,
    total_time,
) -> dict[str, str]:
    return {
        "player1": player1_name,
        "player2": player2_name,
        "winner": winner,
        "win_method": win_method,
        "player1_move_time": player_1_move_time,
        "player2_move_time": player_2_move_time,
        "player1_turns": player_1_turn,
        "player2_turns": player_2_turn,
        "total_turns": total_turns,
        "total_game_time": total_time,
    }


class Game:
    """This class describes a game of Hex."""

    # the maximum time allocated for a match per player
    # 5 minutes in nanoseconds (min * s/min * ns/s)
    MAXIMUM_TIME = 5 * 60 * 10**9
    # 1 second in nanoseconds
    # MAXIMUM_TIME = 10**9

    has_swapped: bool
    players: dict[Colour, Player]
    _turn: int
    logDest: str | TextIO

    def __init__(
        self,
        player1: Player,
        player2: Player,
        board_size: int = 11,
        logDest: str | TextIO = sys.stderr,
        verbose: bool = False,
        silent: bool = False,
    ):
        self._turn = 0  # current turn count
        self._board = Board(board_size)
        self.current_player = Colour.RED  # current player
        self._start_time = time()  # used to calculate time elapsed
        self.has_swapped = False  # pie rule
        self.player1 = player1
        self.player2 = player2

        self.players = {
            Colour.RED: self.player1,
            Colour.BLUE: self.player2,
        }
        # logger.setLevel(logging.DEBUG)

        if verbose:
            logger.setLevel(logging.DEBUG)

        if silent:
            logger.setLevel(logging.CRITICAL)
            logDest = os.devnull

        if logDest != sys.stderr:
            self.logDest = open(logDest, "w")
        else:
            self.logDest = logDest

    @property
    def turn(self):
        return self._turn

    @property
    def board(self):
        return self._board

    def run(self):
        """Runs the match."""
        try:
            assert issubclass(type(self.players[Colour.RED].agent), AgentBase)
            assert issubclass(type(self.players[Colour.BLUE].agent), AgentBase)
            logger.info("Game started")
            return self._play()
        except Exception as e:
            self._end_game(None)
            print(f"Exception raised: {e}")
        finally:
            if self.logDest != sys.stderr:
                self.logDest.close()

    def _play(self) -> dict[str, str]:
        """Main method for a match.

        The engine will keep sending status messages to agents and
        prompting them for moves until one of the following is met:

        * Win - one of the agents connects their sides of the board.
        * Illegal move - one of the agents sends an illegal message.
        * Timeout - one of the agents fails to send a message before
        the time elapses. This can also be prompted if the agent
        fails to connect.
        """
        endState = EndState.WIN
        opponentMove = None

        while True:
            self._turn += 1
            currentPlayer: Player = self.players[self.current_player]
            playerAgent = currentPlayer.agent
            logger.info(f"Turn {self.turn}: player {currentPlayer.name}")
            logger.info(f"Starting Board:\n{str(self.board)}")
            currentPlayer.turn += 1

            boardCopy = copy.deepcopy(self.board)
            turnCopy = self.turn
            playerCopy = copy.deepcopy(self.players)

            playerBoard = copy.deepcopy(self.board)

            start = time()
            m = playerAgent.make_move(self.turn, playerBoard, opponentMove)
            end = time()

            assert boardCopy == self.board, "Board was modified, Possible cheating!"
            assert turnCopy == self.turn, "Turn was modified, Possible cheating!"
            assert (
                playerCopy == self.players
            ), "Players were modified, Possible cheating!"
            assert end > start, "Move time is negative, Possible cheating!"

            currentPlayer.move_time += end - start
            logger.debug(
                f"Player {currentPlayer.name}; Move time: {currentPlayer.move_time}ns"
            )
            logger.info(f"Player {currentPlayer.name}; Move: {self.current_player}{m}")
            if currentPlayer.move_time > Game.MAXIMUM_TIME:
                logger.info(f"Player {currentPlayer.name} timed out")
                endState = EndState.TIMEOUT
                break
            if self.is_valid_move(m, self.turn, self.board):
                logger.debug("Move is valid")
                self._make_move(m)
                opponentMove = m
            else:
                logger.info(f"Player {currentPlayer.name} made an illegal move")
                endState = EndState.BAD_MOVE
                break
            if self.board.has_ended(self.current_player):
                break

            logger.info(f"Turn Ending Board:\n{str(self.board)}")

            self.current_player = Colour.opposite(self.current_player)
        return self._end_game(endState)

    def _make_move(self, m: Move):
        """Performs a valid move on the board, then prints its results."""

        if m.x == -1 and m.y == -1:
            self.players[Colour.RED], self.players[Colour.BLUE] = (
                self.players[Colour.BLUE],
                self.players[Colour.RED],
            )
            self.players[Colour.RED].agent.colour = Colour.RED
            self.players[Colour.BLUE].agent.colour = Colour.BLUE
            logger.debug("This is a swap move.")
            logger.debug(
                f"{self.players[Colour.RED].name} colour:{self.players[Colour.RED].agent.colour.name}"
            )
            logger.debug(
                f"{self.players[Colour.BLUE].name}, colour:{self.players[Colour.BLUE].agent.colour.name}"
            )

            self.current_player = Colour.opposite(self.current_player)
            self.has_swapped = True
        else:
            self.board.set_tile_colour(m.x, m.y, self.current_player)

        logger.debug(f"Move made: {m}")
        current_player = self.players[self.current_player]
        print(
            f"{self.turn},{current_player.name},{self.current_player.name}{m},{current_player.move_time}",
            file=self.logDest,
        )

    def _end_game(self, status: EndState) -> dict[str, str]:
        """Wraps up the game and prints results to shell, log and
        agents.
        """
        # calculate total time elapsed
        total_time = time() - self._start_time

        logger.info("Game over")
        logger.info(f"Final Board:\n{str(self.board)}")
        logger.info(f"Total time: {Game.ns_to_s(total_time)}s")
        winner = None

        match status:
            case EndState.WIN:
                # last move overcounts
                logger.info(f"Player {self.players[self.current_player].name} has won")
                winner = self.players[self.current_player].name
            case EndState.BAD_MOVE:
                # the player printed is the winner
                logger.info(
                    f"Player {self.players[self.current_player].name} made an illegal move"
                )
                logger.info(
                    f"Player {self.players[Colour.opposite(self.current_player)].name} has won"
                )
                winner = self.players[Colour.opposite(self.current_player)].name

            case EndState.TIMEOUT:
                # the player printed is the winner
                # last move overcounts
                logger.info(
                    f"Player {self.players[self.current_player].name} has timed out"
                )
                logger.info(
                    f"Player {self.players[Colour.opposite(self.current_player)].name} has won"
                )
                winner = self.players[Colour.opposite(self.current_player)].name
            case _:
                logger.error("Game ended abnormally")
                raise Exception("Game ended abnormally")

        for p in self.players.values():
            print(f"{p.name},{p.move_time}", file=self.logDest)
        print(f"winner,{winner},{status.name}", file=self.logDest)
        logger.info(f"Total Game Time: {Game.ns_to_s(total_time)}s")

        return format_result(
            player1_name=self.player1.name,
            player2_name=self.player2.name,
            winner=winner,
            win_method=status.name,
            player_1_move_time=Game.ns_to_s(self.player1.move_time),
            player_2_move_time=Game.ns_to_s(self.player2.move_time),
            player_1_turn=self.player1.turn,
            player_2_turn=self.player2.turn,
            total_turns=self._turn,
            total_time=Game.ns_to_s(total_time),
        )

    def is_valid_move(self, move: Move, turn: int, board: Board) -> bool:
        """Checks if the move can be made by the given player at the given
        position.
        """
        if not isinstance(move, Move):
            return False

        if type(move) is not type(Move(0, 0)):
            return False

        if (0 <= move.x < board.size) and (0 <= move.y < board.size):
            # is in bound?
            tile = board.tiles[move.x][move.y]
            return tile.colour is None
        elif move.x == -1 and move.y == -1 and turn == 2:
            # is a swap move?
            return True
        else:
            return False

    @staticmethod
    def ns_to_s(t):
        """Method for standardised nanosecond to second conversion."""
        return int(t / 10**6) / 10**3
