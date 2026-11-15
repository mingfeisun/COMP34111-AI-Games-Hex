import threading
import unittest

from agents.DefaultAgents.NaiveAgent import NaiveAgent
from src.Board import Board
from src.Game import Game
from src.Player import Player
from src.Colour import Colour
from src.Move import Move


class TestNaiveAgent(unittest.TestCase):
    def test_moves_only_to_empty_tiles(self):
        board = Board(11)
        for x in range(11):
            for y in range(11):
                if (x, y) != (5, 7):
                    board.set_tile_colour(x, y, Colour.RED)
        agent = NaiveAgent(Colour.BLUE)
        for _ in range(20):
            self.assertEqual(agent.make_move(5, board, Move(0, 0)), Move(5, 7))

    def test_swaps_only_on_turn_2(self):
        agent = NaiveAgent(Colour.BLUE)
        board = Board(11)
        board.set_tile_colour(0, 0, Colour.RED)
        moves = {agent.make_move(2, board, Move(0, 0)) for _ in range(200)}
        self.assertIn(Move(-1, -1), moves)           # swaps sometimes (50%)
        self.assertTrue(len(moves) > 1)              # and sometimes plays
        for _ in range(50):
            self.assertNotEqual(agent.make_move(4, board, Move(0, 0)), Move(-1, -1))

    def test_hash_same_for_same_class(self):
        self.assertEqual(hash(NaiveAgent(Colour.RED)), hash(NaiveAgent(Colour.BLUE)))



class LockAgent(NaiveAgent):
    """An agent that holds a lock: a deep copy of it raises a TypeError."""

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.lock = threading.Lock()


class TestGameWithAgents(unittest.TestCase):
    def test_agent_with_lock_can_play(self):
        game = Game(Player("A", LockAgent(Colour.RED)), Player("B", NaiveAgent(Colour.BLUE)), silent=True)
        result = game.run()
        self.assertIn(result["winner"], ("A", "B"))
        self.assertEqual(result["win_method"], "WIN")


if __name__ == "__main__":
    unittest.main()
