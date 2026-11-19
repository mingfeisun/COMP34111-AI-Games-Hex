"""Tests for agents in their own process (src/AgentProxy.py, src/AgentHost.py).

The test agents are in this file; the host imports them as
"test.test_agent_proxy <ClassName>". The tests use short limits, so they
take about 20 s. They need Linux (pausing and /proc).
"""

import os
import random
import subprocess
import sys
import tempfile
import time
import unittest

from src.AgentBase import AgentBase
from src.AgentProxy import (
    AgentCrash,
    AgentLoadError,
    AgentProxy,
    AgentTimeout,
    pausing_supported,
)
from src.Board import Board
from src.Colour import Colour
from src.EndState import EndState
from src.Game import Game
from src.Move import Move
from src.Player import Player

HERE = "test.test_agent_proxy"


def first_empty(board):
    for row in board.tiles:
        for t in row:
            if t.colour is None:
                return Move(t.x, t.y)


class QuickAgent(AgentBase):
    def __init__(self, colour):
        super().__init__(colour)

    def make_move(self, turn, board, opp_move):
        return first_empty(board)


class SlowInitAgent(QuickAgent):
    def __init__(self, colour):
        super().__init__(colour)
        time.sleep(3)


class InitErrorAgent(QuickAgent):
    def __init__(self, colour):
        super().__init__(colour)
        raise RuntimeError("cannot find my weights")


class SleepMoveAgent(QuickAgent):
    def make_move(self, turn, board, opp_move):
        time.sleep(5)
        return first_empty(board)


class CrashMoveAgent(QuickAgent):
    def make_move(self, turn, board, opp_move):
        raise ValueError("bug in make_move")


class SubclassMove(Move):
    pass


class SubclassMoveAgent(QuickAgent):
    def make_move(self, turn, board, opp_move):
        m = first_empty(board)
        return SubclassMove(m.x, m.y)


class PrintAgent(QuickAgent):
    def make_move(self, turn, board, opp_move):
        print("debug text on stdout")
        return first_empty(board)


class ThreadAgent(QuickAgent):
    """Starts a thread that appends to a file every 10 ms (pondering)."""

    def __init__(self, colour):
        super().__init__(colour)
        import threading

        path = os.environ["TEST_TICK_FILE"]

        def tick():
            while True:
                with open(path, "a") as f:
                    f.write(".")
                time.sleep(0.01)

        threading.Thread(target=tick, daemon=True).start()


class EscapeAgent(QuickAgent):
    """Starts a busy process that leaves the process group (setsid)."""

    def __init__(self, colour):
        super().__init__(colour)
        p = subprocess.Popen([sys.executable, "-c", "while True: pass"], start_new_session=True)
        with open(os.environ["TEST_PID_FILE"], "w") as f:
            f.write(str(p.pid))


class MemoryAgent(QuickAgent):
    def __init__(self, colour):
        super().__init__(colour)
        self.block = bytearray(300 * 2**20)  # 300 MB, written so that it is resident
        for i in range(0, len(self.block), 4096):
            self.block[i] = 1


class SeedAgent(QuickAgent):
    def make_move(self, turn, board, opp_move):
        return Move(random.randrange(board.size), random.randrange(board.size))


class ColourCheckAgent(QuickAgent):
    """Player 1. After the opponent swaps on turn 2, it must be Blue."""

    def make_move(self, turn, board, opp_move):
        if turn == 3 and opp_move is not None and opp_move.is_swap():
            assert self.colour == Colour.BLUE, f"colour is {self.colour} after the swap"
        return first_empty(board)


@unittest.skipUnless(pausing_supported() and os.path.isdir("/proc/self"), "needs Linux")
class TestAgentProxy(unittest.TestCase):
    def setUp(self):
        self.proxies = []

    def tearDown(self):
        for p in self.proxies:
            p.close()

    def proxy(self, cls, colour=Colour.RED, **kw):
        p = AgentProxy(f"{HERE} {cls}", colour, 11, **kw)
        self.proxies.append(p)
        return p

    def test_move_and_close(self):
        p = self.proxy("QuickAgent")
        self.assertLess(p.startup_s, 5)
        self.assertEqual(p.make_move(1, Board(11), None), Move(0, 0))
        p.close()
        self.assertIsNotNone(p.process.poll())

    def test_startup_limit(self):
        with self.assertRaises(AgentLoadError) as cm:
            self.proxy("SlowInitAgent", startup_limit_s=1)
        self.assertIn("start-up", str(cm.exception))

    def test_init_error(self):
        with self.assertRaises(AgentLoadError) as cm:
            self.proxy("InitErrorAgent")
        self.assertIn("cannot find my weights", str(cm.exception))

    def test_import_error(self):
        with self.assertRaises(AgentLoadError):
            AgentProxy("agents.NoSuchGroup.NoSuchModule Agent", Colour.RED)

    def test_timeout_during_move(self):
        p = self.proxy("SleepMoveAgent", time_limit_s=1)
        start = time.perf_counter()
        with self.assertRaises(AgentTimeout):
            p.make_move(1, Board(11), None)
        self.assertLess(time.perf_counter() - start, 4)

    def test_crash(self):
        p = self.proxy("CrashMoveAgent")
        with self.assertRaises(AgentCrash) as cm:
            p.make_move(1, Board(11), None)
        self.assertIn("bug in make_move", cm.exception.agent_traceback)

    def test_move_subclass_is_rejected(self):
        p = self.proxy("SubclassMoveAgent")
        self.assertIsNone(p.make_move(1, Board(11), None))

    def test_print_does_not_break_protocol(self):
        p = self.proxy("PrintAgent")
        self.assertEqual(p.make_move(1, Board(11), None), Move(0, 0))

    def test_threads_are_paused(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tick = f.name
        os.environ["TEST_TICK_FILE"] = tick
        try:
            p = self.proxy("ThreadAgent")
            p.make_move(1, Board(11), None)
            time.sleep(0.2)
            size = os.path.getsize(tick)
            time.sleep(1.0)
            self.assertEqual(os.path.getsize(tick), size, "the thread ran while paused")
            p.make_move(3, Board(11), Move(5, 5))
            time.sleep(0.2)
            self.assertEqual(p.finish_checks(), [])
        finally:
            os.unlink(tick)

    def test_escaped_process_is_logged_and_killed(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pid_file = f.name
        os.environ["TEST_PID_FILE"] = pid_file
        try:
            p = self.proxy("EscapeAgent")
            time.sleep(1.6)  # paused; the escaped process keeps running
            warnings = p.finish_checks()
            self.assertTrue(any("while the agent was paused" in w for w in warnings), warnings)
            with open(pid_file) as f:
                pid = int(f.read())
            p.close()
            time.sleep(0.2)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    state = f.read().split(")")[-1].split()[0]
            except OSError:
                state = "gone"
            self.assertIn(state, ("gone", "Z"))
        finally:
            os.unlink(pid_file)

    def test_memory_check(self):
        p = self.proxy("MemoryAgent", memory_limit_bytes=100 * 2**20)
        time.sleep(1.2)
        self.assertGreater(p.peak_memory_bytes, 250 * 2**20)
        self.assertTrue(any("peak memory" in w for w in p.finish_checks()))

    def test_seed(self):
        a = self.proxy("SeedAgent", seed=42)
        b = self.proxy("SeedAgent", seed=42)
        self.assertEqual(a.make_move(1, Board(11), None), b.make_move(1, Board(11), None))

    def test_game_with_swap(self):
        p1 = self.proxy("ColourCheckAgent", Colour.RED)
        p2 = AgentProxy("agents.TestAgents.ValidAgent ValidAgent", Colour.BLUE)
        self.proxies.append(p2)
        result = Game(Player("A", p1), Player("B", p2), silent=True).run()
        self.assertEqual(result["win_method"], EndState.WIN.name)

    def test_game_timeout(self):
        p1 = self.proxy("SleepMoveAgent", Colour.RED, time_limit_s=1)
        p2 = self.proxy("QuickAgent", Colour.BLUE)
        old = Game.MAXIMUM_TIME
        Game.MAXIMUM_TIME = 1 * 10**9
        try:
            result = Game(Player("A", p1), Player("B", p2), silent=True).run()
        finally:
            Game.MAXIMUM_TIME = old
        self.assertEqual(result["win_method"], EndState.TIMEOUT.name)
        self.assertEqual(result["winner"], "B")


if __name__ == "__main__":
    unittest.main()
