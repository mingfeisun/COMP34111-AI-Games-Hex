"""Use this script to check your code before you submit it.

Usage (from the repository root, inside the Docker container):

    python3 check_submission.py 7          # checks agents/Group007/
    python3 check_submission.py Group007   # the same

It checks the folder and cmd.txt in the same way as the marking scripts, then
plays three games with the real time limits:

    1. your bot as player 1 (Red)  against NaiveAgent
    2. NaiveAgent as player 1      against your bot as player 2 (Blue)
    3. your bot as player 1        against ValidAgent, which always swaps on
       turn 2 (your bot gets Move(-1, -1) on turn 3 and is then Blue)

The games run with Hex.py. Each bot runs in its own process and is paused during 
the opponent's turn. The script also shows if your bot tries to use more resources 
than allowed, or tries to think while it is not its turn. 

Passing these checks does not mean that your bot is strong. It means that it
loads, plays legal moves, handles a swap and stays inside the limits. In other words,
it fulfills the minimum requirements for a working submission.
"""

import os
import re
import subprocess
import sys
import tempfile
import time

MAX_SIZE_BYTES = 1 * 1024**3         # 1 GB for GroupXYZ.tar
MOVE_TIME_LIMIT_S = 5 * 60           # 5 minute chess clock per player per game
STARTUP_LIMIT_S = 3 * 60             # agent __init__ gets 3 minutes
GAME_WALL_LIMIT_S = 2 * (MOVE_TIME_LIMIT_S + STARTUP_LIMIT_S) + 30
NAIVE = "agents.DefaultAgents.NaiveAgent NaiveAgent"
SWAPPER = "agents.TestAgents.ValidAgent ValidAgent"

problems = []
warnings = []


def fail(msg):
    problems.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARNING: {msg}")


def ok(msg):
    print(f"  ok: {msg}")


def folder_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def check_folder(group):
    print(f"\n[1] Folder agents/{group}/")
    path = os.path.join("agents", group)
    if not re.fullmatch(r"Group\d{3}", group):
        fail(f"folder name '{group}' must be 'Group' + your Canvas group number "
             "with three digits, e.g. Group007")
    if not os.path.isdir(path):
        fail(f"{path} does not exist (run this script from the repository root)")
        return None
    size = folder_size(path)
    if size > MAX_SIZE_BYTES:
        fail(f"folder is {size / 2**20:.0f} MB; the maximum is {MAX_SIZE_BYTES / 2**20:.0f} MB")
    else:
        ok(f"size {size / 2**20:.1f} MB (maximum {MAX_SIZE_BYTES / 2**20:.0f} MB)")
    bundled = []
    for root, dirs, _files in os.walk(path):
        for d in dirs:
            if d.endswith((".dist-info", ".egg-info")) or d in ("site-packages", "venv", ".venv"):
                bundled.append(os.path.join(root, d))
    if bundled:
        warn("your folder seems to contain copies of libraries (do not bundle libraries): "
             + ", ".join(bundled[:5]))
    return path


def check_cmd(group, path):
    print("\n[2] cmd.txt")
    cmd_path = os.path.join(path, "cmd.txt")
    if not os.path.isfile(cmd_path):
        fail(f"{cmd_path} does not exist")
        return None
    raw = open(cmd_path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"):
        fail("cmd.txt starts with a byte order mark (BOM); save it as UTF-8 without BOM")
    text = raw.decode("utf-8", "replace")
    # the marking scripts read the first line only, and strip spaces and \r
    agent = text.split("\n")[0].strip()
    others = [line for line in text.split("\n")[1:] if line.strip()]
    if others:
        warn("cmd.txt has more than one line; only the first line is used")
    fields = agent.split(" ")
    if len(fields) != 2:
        fail(f"the line must be '<module> <ClassName>' with exactly one space; found {agent!r}")
        return None
    module, cls = fields
    parts = module.split(".")
    if len(parts) < 3 or parts[0] != "agents" or parts[1] != group:
        fail(f"the module must start with 'agents.{group}.'; found {module!r} "
             "(the group name must match the folder name exactly)")
        return None
    mod_file = os.path.join(*parts) + ".py"
    if not (os.path.isfile(mod_file) or os.path.isfile(os.path.join(*parts, "__init__.py"))):
        fail(f"{mod_file} does not exist")
        return None
    ok(f"{agent}")
    return agent


def check_load(agent):
    print("\n[3] Load and start-up")
    module, cls = agent.split(" ")
    code = (
        "import importlib, time\n"
        "from src.AgentBase import AgentBase\n"
        "from src.Colour import Colour\n"
        f"t = time.perf_counter()\n"
        f"m = importlib.import_module({module!r})\n"
        f"c = getattr(m, {cls!r})\n"
        "assert issubclass(c, AgentBase), 'the class must inherit from AgentBase'\n"
        "a = c(Colour.RED)\n"
        "print('STARTUP_SECONDS', time.perf_counter() - t)\n"
    )
    start = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=STARTUP_LIMIT_S + 30)
    except subprocess.TimeoutExpired:
        fail(f"loading and start-up took more than {STARTUP_LIMIT_S} s")
        return False
    m = re.search(r"STARTUP_SECONDS ([0-9.]+)", r.stdout)
    if r.returncode != 0 or not m:
        fail("the agent could not be imported or constructed:\n" + r.stderr.strip()[-2000:])
        return False
    secs = float(m.group(1))
    if secs > STARTUP_LIMIT_S:
        fail(f"start-up took {secs:.1f} s; the limit is {STARTUP_LIMIT_S} s")
    elif secs > STARTUP_LIMIT_S / 2:
        warn(f"start-up took {secs:.1f} s; the limit is {STARTUP_LIMIT_S} s")
    else:
        ok(f"import and __init__ took {secs:.1f} s (limit {STARTUP_LIMIT_S} s)")
    return True


def play(p1, p2, bot_name, title):
    print(f"\n[game] {title}")
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "game.log")
        n1 = "Bot" if bot_name == "p1" else "Opponent"
        n2 = "Bot" if bot_name == "p2" else "Opponent"
        try:
            r = subprocess.run([sys.executable, "Hex.py", "-p1", p1, "-p1Name", n1, "-p2", p2,
                                "-p2Name", n2, "-l", log],
                               capture_output=True, text=True, timeout=GAME_WALL_LIMIT_S)
        except subprocess.TimeoutExpired:
            fail(f"the game did not finish in {GAME_WALL_LIMIT_S} s (does your bot hang?)")
            return
        lines = open(log).read().splitlines() if os.path.exists(log) else []
        result = [l for l in lines if l.startswith("winner,")]
        if not result:
            fail("the game ended without a result:\n" + (r.stdout + r.stderr).strip()[-2000:])
            return
        _, winner, state = result[-1].split(",")
        times = {l.split(",")[0]: int(float(l.split(",")[1])) for l in lines
                 if l.split(",")[0] in ("Bot", "Opponent") and len(l.split(",")) == 2}
        bot_s = times.get("Bot", 0) / 1e9
        for l in lines:
            if l.startswith("# WARNING Bot:"):
                warn("rule check: " + l[len("# WARNING Bot:"):].strip()
                     + " (see 'Rules in marked games' in docs/GAME.md)")
        if winner != "Bot" and state != "WIN":
            fail(f"your bot lost by {state}")
            crash = [l[2:] for l in lines if l.startswith("# ")]
            if crash:
                print("    " + "\n    ".join(crash[-15:]))
        else:
            ok(f"winner {winner} by {state}")
        if bot_s > MOVE_TIME_LIMIT_S * 0.5:
            warn(f"your bot used {bot_s:.1f} s of its {MOVE_TIME_LIMIT_S} s; leave a margin")
        else:
            ok(f"your bot used {bot_s:.1f} s of its {MOVE_TIME_LIMIT_S} s")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    arg = sys.argv[1].strip().rstrip("/\\")
    group = os.path.basename(arg) if not arg.isdigit() else f"Group{int(arg):03d}"
    if not os.path.isfile("Hex.py"):
        print("Run this script from the repository root.")
        sys.exit(2)
    path = check_folder(group)
    agent = check_cmd(group, path) if path else None
    if agent and check_load(agent):
        play(agent, NAIVE, "p1", "your bot (player 1, Red) vs NaiveAgent")
        play(NAIVE, agent, "p2", "NaiveAgent (player 1, Red) vs your bot (player 2, Blue)")
        play(agent, SWAPPER, "p1", "your bot (player 1) vs ValidAgent, which swaps on turn 2")
    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} problem(s) found. Fix them before you submit.")
        sys.exit(1)
    print(f"All checks passed ({len(warnings)} warning(s)).")


if __name__ == "__main__":
    main()
