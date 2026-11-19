"""Runs one agent in its own process. The engine starts it with AgentProxy.

    python3 -m src.AgentHost <module> <ClassName> <R|B> <board_size>

The engine and this host talk over two pipes that are not stdin or stdout
(their file descriptors are in HEX_CMD_FD and HEX_REPLY_FD). So anything the
agent prints goes to the terminal and does not affect the game.

Engine -> host: the message format in docs/GAME.md, COMMAND;MOVE;BOARD;TURN;
Host -> engine: one JSON object per line:
    {"ready": <start-up seconds>}   after __init__
    {"move": [x, y]}                the agent's move
    {"bad": "<text>"}               make_move did not return a Move object
    {"crash": "<traceback>"}        make_move raised an exception
    {"error": "<traceback>"}        import or __init__ failed

The engine pauses this process (and every process it starts) with SIGSTOP
during the opponent's turn, and resumes it with SIGCONT.
"""

import importlib
import json
import os
import random
import sys
import time
import traceback


def _seed_before_import(seed):
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % 2**32)
    except Exception:
        pass


def _seed_after_import(seed):
    # torch is slow to import, so it is seeded only if the agent imported it
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass


def _board_from_protocol(text, size):
    from src.Board import Board
    from src.Colour import Colour

    board = Board(size)
    for x, row in enumerate(text.split(",")):
        for y, c in enumerate(row):
            colour = Colour.from_char(c)
            if colour is not None:
                board.set_tile_colour(x, y, colour)
    return board


def main():
    module, cls_name, colour_char, size = sys.argv[1:5]
    size = int(size)
    cmd_in = os.fdopen(int(os.environ["HEX_CMD_FD"]), "r")
    reply = os.fdopen(int(os.environ["HEX_REPLY_FD"]), "w", buffering=1)

    def send(obj):
        reply.write(json.dumps(obj) + "\n")
        reply.flush()

    start = time.perf_counter()
    seed = os.environ.get("HEX_SEED")
    seed = int(seed) if seed not in (None, "") else None
    try:
        if seed is not None:
            _seed_before_import(seed)
        from src.AgentBase import AgentBase
        from src.Colour import Colour
        from src.Move import Move

        cls = getattr(importlib.import_module(module), cls_name)
        if not (isinstance(cls, type) and issubclass(cls, AgentBase)):
            raise TypeError(f"{module}.{cls_name} does not inherit from AgentBase")
        if seed is not None:
            _seed_after_import(seed)
        agent = cls(Colour.from_char(colour_char))
    except BaseException:
        send({"error": traceback.format_exc()})
        return 1
    send({"ready": time.perf_counter() - start})

    for line in cmd_in:
        parts = line.rstrip("\n").split(";")
        command, move_text, board_text, turn = parts[0], parts[1], parts[2], int(parts[3])
        board = _board_from_protocol(board_text, size)
        if command == "START":
            opp_move = None
        elif command == "SWAP":
            opp_move = Move(-1, -1)
            # as in the in-process engine: the opponent swapped, so our colour changes
            agent.colour = Colour.opposite(agent.colour)
        else:
            x, y = move_text.split(",")
            opp_move = Move(int(x), int(y))
        try:
            m = agent.make_move(turn, board, opp_move)
        except BaseException:
            send({"crash": traceback.format_exc()})
            return 1
        # the engine accepts only a plain Move object (no subclass)
        if type(m) is not Move:
            send({"bad": f"make_move returned {type(m).__name__}, not Move"})
            continue
        if m.is_swap() and turn == 2:
            # our own swap: we play the other colour from now on
            agent.colour = Colour.opposite(agent.colour)
        send({"move": [m.x, m.y]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
