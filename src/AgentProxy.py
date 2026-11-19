"""Engine side of an agent that runs in its own process (see AgentHost.py).

Each AgentProxy:
- starts the agent in a new process group and waits for its start-up
  (import and __init__), with a time limit;
- pauses the whole process group with SIGSTOP when it is not the agent's turn,
  and resumes it with SIGCONT for its move;
- stops waiting for a move when the agent's clock has run out (AgentTimeout);
- kills the process group at the end of the game.

A monitor thread reads /proc (Linux only) every 0.5 s. It records rule
breaks, which the game log shows as "# WARNING" lines:
- a process of the agent that uses CPU time while the agent is paused (for
  example a process that left the process group with setsid);
- the agent's peak memory (sum over its processes), if it is above the limit.
Every process of the agent has the environment variable HEX_AGENT, so the
monitor can find processes that left the process group.

Pausing needs POSIX signals: it works on Linux (the Docker image) and macOS,
not on Windows.
"""

import atexit
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

STARTUP_LIMIT_S = 3 * 60
MOVE_TIME_LIMIT_S = 5 * 60
MEMORY_LIMIT_BYTES = 8 * 1024**3
MONITOR_INTERVAL_S = 0.5
# The engine (Game.py) decides timeouts from its own clock. The proxy waits a
# little longer, so that it only stops waiting when the move is certainly late.
_WAIT_MARGIN_S = 1.0


class AgentLoadError(Exception):
    """The agent could not be imported or constructed, or its start-up was too slow."""


class AgentTimeout(Exception):
    """The agent used all its time during a move."""


class AgentCrash(Exception):
    """The agent raised an exception, or its process stopped, during a move."""

    def __init__(self, message, agent_traceback=None):
        super().__init__(message)
        self.agent_traceback = agent_traceback


def pausing_supported() -> bool:
    return hasattr(os, "killpg") and hasattr(signal, "SIGSTOP")


class AgentProxy(AgentBase):
    def __init__(
        self,
        agent: str,
        colour: Colour,
        board_size: int = 11,
        *,
        seed: int | None = None,
        startup_limit_s: float = STARTUP_LIMIT_S,
        time_limit_s: float = MOVE_TIME_LIMIT_S,
        memory_limit_bytes: int = MEMORY_LIMIT_BYTES,
        cwd: str | None = None,
        user: str | int | None = None,
        group: str | int | None = None,
    ):
        """agent: "<module> <ClassName>", as in cmd.txt.
        user, group: run the agent as this user (the engine must be root).
        """
        super().__init__(colour)
        self.agent_spec = agent
        self.board_size = board_size
        self.time_limit_s = time_limit_s
        self.memory_limit_bytes = memory_limit_bytes
        self.used_s = 0.0
        self.startup_s = None
        self.peak_memory_bytes = 0
        self.rule_warnings: list[str] = []
        self._paused = False
        self._closed = False
        self._buf = b""
        self._marker = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pause_seq = 0
        self._baseline: dict[int, int] = {}  # CPU ticks of escaped processes at the pause
        self._baseline_seq = -1
        self._warned_pids: set[int] = set()

        module, cls_name = agent.split(" ")
        cmd_r, cmd_w = os.pipe()
        reply_r, reply_w = os.pipe()
        env = dict(os.environ)
        env.update(
            HEX_AGENT=self._marker,
            HEX_CMD_FD=str(cmd_r),
            HEX_REPLY_FD=str(reply_w),
            HEX_SEED="" if seed is None else str(seed),
            PYTHONUNBUFFERED="1",
        )
        kwargs = {}
        if user is not None:
            kwargs.update(user=user, group=group, extra_groups=[])
        start = time.perf_counter()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "src.AgentHost", module, cls_name,
             Colour.to_protocol_char(colour), str(board_size)],
            stdin=subprocess.DEVNULL,
            pass_fds=(cmd_r, reply_w),
            env=env,
            cwd=cwd,
            start_new_session=True,  # own process group, for pausing
            **kwargs,
        )
        os.close(cmd_r)
        os.close(reply_w)
        self._cmd = os.fdopen(cmd_w, "w", buffering=1)
        self._reply_fd = reply_r
        self.pgid = self.process.pid
        atexit.register(self.close)

        self._monitor = None
        if os.path.isdir("/proc/self"):
            self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor.start()

        msg = self._read_message(start + startup_limit_s)
        if msg is None:
            self.close()
            raise AgentLoadError(
                f"start-up (import and __init__) took more than {startup_limit_s:.0f} s")
        if "error" in msg:
            self.close()
            raise AgentLoadError(msg["error"])
        self.startup_s = time.perf_counter() - start
        self._pause()

    # --- the game ---------------------------------------------------------

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move | None:
        board_string = board.to_protocol_string()
        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.is_swap():
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        start = time.perf_counter()
        deadline = start + max(0.0, self.time_limit_s - self.used_s) + _WAIT_MARGIN_S
        self._resume()
        try:
            self._cmd.write(command + "\n")
            self._cmd.flush()
            msg = self._read_message(deadline)
        except (BrokenPipeError, OSError):
            msg = {}
        finally:
            self._pause()
        self.used_s += time.perf_counter() - start

        if msg is None:
            raise AgentTimeout(f"no move after {self.used_s:.1f} s in total")
        if "move" in msg:
            x, y = msg["move"]
            return Move(int(x), int(y))
        if "bad" in msg:
            return None  # the engine treats this as an illegal move
        if "crash" in msg:
            raise AgentCrash("the agent raised an exception", msg["crash"])
        code = self.process.poll()
        raise AgentCrash(f"the agent's process stopped (exit code {code}) without a move")

    # --- process control --------------------------------------------------

    def _signal(self, sig):
        try:
            os.killpg(self.pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _pause(self):
        with self._lock:
            self._signal(signal.SIGSTOP)
            self._paused = True
            self._pause_seq += 1
        self._wake.set()  # the monitor takes a baseline now, in its own thread

    def _resume(self):
        with self._lock:
            # processes that left the process group are not stopped by SIGSTOP:
            # did they use CPU time during this pause?
            if self._baseline_seq == self._pause_seq:
                for pid, before in self._baseline.items():
                    self._check_cpu(pid, before)
            self._paused = False
            self._signal(signal.SIGCONT)

    def _check_cpu(self, pid, before):
        st = self._proc_stat(pid)
        if st is None:
            return
        _state, _pgrp, cpu, comm = st
        if cpu > before and pid not in self._warned_pids:
            self._warned_pids.add(pid)
            self.rule_warnings.append(
                f"process {pid} ({comm}) used CPU time while the agent was paused")

    def _read_message(self, deadline):
        """Returns the next JSON message, {} at end of file, or None at the deadline."""
        while b"\n" not in self._buf:
            left = deadline - time.perf_counter()
            if left <= 0:
                return None
            ready, _, _ = select.select([self._reply_fd], [], [], left)
            if not ready:
                return None
            chunk = os.read(self._reply_fd, 65536)
            if not chunk:
                return {}
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        try:
            return json.loads(line)
        except ValueError:
            return {}

    def close(self):
        """Kills the agent and every process it started. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._paused = False
        self._signal(signal.SIGKILL)
        for pid in self._agent_pids():  # processes that left the process group
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            self.process.wait(timeout=5)
        except Exception:
            pass
        for f in (self._cmd,):
            try:
                f.close()
            except Exception:
                pass
        try:
            os.close(self._reply_fd)
        except OSError:
            pass

    # --- monitor (rule checks) -------------------------------------------

    def _agent_pids(self):
        pids = []
        marker = f"HEX_AGENT={self._marker}".encode()
        for name in os.listdir("/proc") if os.path.isdir("/proc") else []:
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/environ", "rb") as f:
                    if marker in f.read().split(b"\0"):
                        pids.append(int(name))
            except OSError:
                continue
        return pids

    @staticmethod
    def _proc_stat(pid):
        """Returns (state, process group, CPU ticks, command name) or None."""
        try:
            with open(f"/proc/{pid}/stat") as f:
                s = f.read()
        except OSError:
            return None
        comm = s[s.index("(") + 1 : s.rindex(")")]
        fields = s[s.rindex(")") + 2 :].split()
        return fields[0], int(fields[2]), int(fields[11]) + int(fields[12]), comm

    @staticmethod
    def _memory(pid):
        """Proportional set size (PSS) in bytes, or the RSS if PSS is not available."""
        for path, key in ((f"/proc/{pid}/smaps_rollup", "Pss:"), (f"/proc/{pid}/status", "VmRSS:")):
            try:
                with open(path) as f:
                    for line in f:
                        if line.startswith(key):
                            return int(line.split()[1]) * 1024
            except OSError:
                continue
        return 0

    def _monitor_loop(self):
        # Wakes at each pause and at least every MONITOR_INTERVAL_S. All work
        # here is in this thread, so it is not counted in the agent's move time.
        while not self._closed:
            self._wake.wait(MONITOR_INTERVAL_S)
            self._wake.clear()
            if self._closed:
                return
            seq = self._pause_seq
            pids = self._agent_pids()
            total = sum(self._memory(p) for p in pids)
            if total > self.peak_memory_bytes:
                self.peak_memory_bytes = total
            # processes of this agent outside its process group
            stats = {pid: self._proc_stat(pid) for pid in pids}
            escaped = {pid: st for pid, st in stats.items()
                       if st is not None and st[1] != self.pgid}
            with self._lock:
                if not self._paused or self._closed or seq != self._pause_seq:
                    continue
                if self._baseline_seq == seq:
                    # a long pause: check again, then start a new baseline
                    for pid, before in self._baseline.items():
                        self._check_cpu(pid, before)
                self._baseline = {pid: st[2] for pid, st in escaped.items()}
                self._baseline_seq = seq

    def finish_checks(self) -> list[str]:
        """Rule-break warnings for the game log, including the memory check."""
        warnings = list(self.rule_warnings)
        if self.peak_memory_bytes > self.memory_limit_bytes:
            warnings.append(
                f"peak memory {self.peak_memory_bytes / 2**30:.2f} GB is above the limit of "
                f"{self.memory_limit_bytes / 2**30:.0f} GB")
        return warnings
