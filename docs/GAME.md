# Hex Game Engine Documentation

COMP34111 AI & Games. For the Docker commands, see the [README](../README.md).
Before you submit, follow the
[submission checklist](SUBMISSION-CHECKLIST.md). Dates and deadlines are on
Canvas.

- [Submission](#submission)
- [General Information](#general-information)
- [Implementation](#implementation)
  - [Overview](#overview) · [Python agent](#python-agent) ·
    [Alternative languages](#alternative-languages) ·
    [Message format](#message-format) ·
    [Compiled code and build.sh](#compiled-code-and-buildsh)
- [Running the game](#running-the-game)
  - [From Terminal](#from-terminal) · [Docker](#docker) ·
    [Log format](#log-format) · [cmd.txt](#cmdtxt)
- [Rules in marked games](#rules-in-marked-games)
- [How your bot is marked](#how-your-bot-is-marked)

## Submission

Submit one file per group, `GroupXYZ.tar`, where XYZ is your Canvas group
number with three digits (for example `Group007.tar`). It contains your group
folder, `GroupXYZ/`, at the top level. Make it from inside the `agents`
folder, for example `tar -cvf Group007.tar Group007` (this command also works
in Windows Command Prompt and PowerShell). The maximum size is 1 GB.

The folder contains your Python agent class (see [Implementation](#implementation)),
its `cmd.txt` (see [cmd.txt](#cmdtxt)), every other file that your agent
needs, and, if your agent has compiled parts, their source and a `build.sh`
(see [Compiled code and build.sh](#compiled-code-and-buildsh)). Do not include
copies of libraries: use only the packages in the Docker image.

**This assessment is AI prohibited:** all submissions must be written by
humans (see the coursework brief).

Before you submit, follow the [submission checklist](SUBMISSION-CHECKLIST.md):
it tells you how to test your bot in the container with
`check_submission.py`, and how to test your `.tar` file on a fresh copy of
the repository.

**IMPORTANT:** We do not accept fixes for broken bots after the deadline. However,
we will test your bot for you before the deadline. If you submit **before 14 December 14:00**
we will tell you if your bot runs within the time limit, and if we can rebuild its 
compiled parts with its `build.sh`. We will not tell you its speed or 
performance. You may change your submission after this and resubmit as many times
as you like, but we will not test any submissions after 14 December 14:00. 
**It is your responsibility to ensure your bot runs and builds correctly ahead of the deadline.**

## General Information

- The rules for Hex can be found
  [here](https://en.wikipedia.org/wiki/Hex_(board_game)).
- The board is 11x11, and it is represented by a 0-indexed two-dimensional
  matrix with rows that fall diagonally to the right. The tile `(x, y)` is in
  row `x` (0 = top) and column `y` (0 = left): in code it is
  `board.tiles[x][y]`, and a move to it is `Move(x, y)`. The tile `(x, y)` has
  the neighbours `(x, y-1)` and `(x, y+1)` in the same row, `(x-1, y)` and
  `(x-1, y+1)` in the row above, and `(x+1, y-1)` and `(x+1, y)` in the row
  below. Below you can see the neighbours of tile X = (1, 2) marked as N:

  | row x \ column y | 0 | 1 | 2 | 3 | 4 |
  | --- | --- | --- | --- | --- | --- |
  | 0 | 0 | 0 | N | N | 0 |
  | 1 | 0 | N | X | N | 0 |
  | 2 | 0 | N | N | 0 | 0 |

- Red aims to connect the top and bottom sides (rows 0 and 10), while Blue
  connects the left and right sides (columns 0 and 10). Red (player 1) moves
  first.
- The pie rule is implemented: on turn 2 only, player 2 (Blue) can swap
  instead of making a normal move. Then the players change colours: player 2
  becomes Red and owns the stone on the board, and player 1 becomes Blue and
  plays turn 3.
- Each agent has a maximum total of 5 minutes per match for all its moves,
  and 3 minutes for its start-up (`__init__`).
- Your agent's performance will be evaluated by a combination of win rate
  (75%) and move speed (25%) (see [How your bot is marked](#how-your-bot-is-marked)).
- Possible outcome of a match:
  - Win (`WIN`) – one agent has connected their sides of the board.
  - Timeout (`TIMEOUT`) – one agent has used up all the time allocated to it
    (5 minutes). The opposing agent wins.
  - Illegal move (`BAD_MOVE`) – one agent tries to occupy an already occupied
    tile, plays a move outside the board, swaps when it is not turn 2, or does
    not return a `Move` object. The opposing agent wins.
  - Crash (`CRASH`) – one agent's `make_move` raises an exception, or its
    process stops. The opposing agent wins.
  - Failed load (`FAILED_LOAD`) – one agent cannot be imported, its
    `__init__` raises an exception, or its start-up takes more than 3
    minutes. The opposing agent wins.

## Implementation

### Overview

You have the option to implement your agent in Python, or in an alternative
programming language of your choice. You can either implement the provided
agent template class in Python, or use the provided
`agents/DefaultAgents/ExternalAgent.py` class to launch your agent as an
external process. More detailed instructions on how to do this are provided
later, with an example implementation in Java.

### Python agent

An agent template is provided at `agents/DefaultAgents/NaiveAgent.py`. The
class has two methods, `__init__` and `make_move`. (`NaiveAgent` plays a
random empty tile, and swaps on turn 2 half of the time.)

When the game starts, the engine will call the `__init__` method in your
Python agent, which will tell you which colour your agent will be playing as
(`Colour.RED` or `Colour.BLUE`) and allow you to perform any setup steps.
Call `super().__init__(colour)`.

Your agent runs in its own process, so it cannot see the engine's or the
opponent's Python objects. Text that your agent prints goes to the terminal
and does not affect the game.

When it is your turn to move, the game engine will call the `make_move`
method, which should always return a `Move` object. If the method does not
return a valid `Move` object, the game engine will classify the move as
illegal, and your agent will lose the game immediately. When the `make_move`
method is called, it gets the turn number (from 1), a copy of the `Board`
specifying the current game state, and a `Move` object specifying the
opponent's move (`None` on turn 1). The `Board` object contains a 2D list of
`Tile` objects, `board.tiles`. `Tile` objects have `x`, `y` and `colour`
properties (`colour` is `None` for an empty tile); `x` and `y` match their
position in `board.tiles`.

The `Move` object can be used to make a swap move or a normal move. To make a
swap move, the `Move` object is created with the parameters `(-1, -1)`. To
make a normal move, the `Move` object is created with the parameters `(x, y)`,
where `x` and `y` are the coordinates the tile will be placed at. If the
opponent swapped, `opp_move` is `Move(-1, -1)`, and the engine has already
changed your `colour` property.

The agent's only implementation requirement is it must be a class that
inherits the `AgentBase` class (located at `src/AgentBase.py`). `AgentBase`
is an abstract base class that ensures you implement both the `__init__` and
`make_move` methods yourself, and provides all the necessary methods for
running a game.

### Alternative languages

The provided `agents/DefaultAgents/ExternalAgent.py` class contains code to
launch an external Java process and communicate with it via stdin and stdout
to play the game. There is a corresponding
`agents/DefaultAgents/NaiveAgent.java` file which demonstrates the
communication for a Java agent implementation. You are free to use or adapt
this example code however you like; your ability to write an interface
between Python and another programming language is not part of the assessment
for this module. You should be able to use this class to run an external
agent in any programming language that can communicate with stdin and stdout,
by changing the command in its `__init__` method:

```python
self.agent_process = Popen(
    ["java", "-cp", "agents/DefaultAgents", "NaiveAgent", colour.to_protocol_char(), "11"],
    stdout=PIPE, stdin=PIPE, text=True,
)
```

The provided example is equivalent to running the command
`java -cp agents/DefaultAgents NaiveAgent B 11` in the terminal, assuming the
agent has been assigned the colour Blue. The `-cp agents/DefaultAgents` part
of the command specifies the location of the compiled `NaiveAgent.class`
file. If you copy the class into your group folder, change this to
`agents/GroupXYZ`, and name the class in your `cmd.txt`. A program that
your agent starts is paused together with your agent during the opponent's
turn (see [Rules in marked games](#rules-in-marked-games)).

If using the provided `ExternalAgent` code, when the game is started your
code will be launched using the command specified in the `__init__` method.
On each of your agent's turns, a message will be sent via your process' stdin
in the format described below, and the game will wait for a response via
your process' stdout. Each message sent by the game will end in a newline
(`\n`). Your response should consist of one line, also ending in a newline.

#### Message format

The basic message format that the game will send to your agent is
`COMMAND;MOVE;BOARD;TURN;\n`. Your agent should respond with `x,y\n`, where
`x` and `y` are the coordinates of the tile you are changing. To make a swap
move, `x` and `y` should both be set to `-1`.

`COMMAND` can take the values `START`, `SWAP`, or `CHANGE`. `START` is only
used on the first turn of the game – i.e. you will only receive a `START`
message if you are Red. `SWAP` indicates that the opponent has chosen to swap
colours: you are now Blue. The swap move is only allowed for player 2 (Blue)
on their first turn – i.e. only on turn 2 of the game. If your own agent
swaps, it is Red from then on; the engine does not send a message about your
own swap. `CHANGE` specifies a tile that the opponent has changed to their
colour. `MOVE` will be blank for `START` and `SWAP` messages; `BOARD` and
`TURN` will always be specified.

`MOVE` takes the value of the `x,y` coordinates of the tile the opponent has
changed when they make a `CHANGE` move.

`BOARD` specifies the state of the board at the beginning of your turn, with
each tile taking the value `0`, `R` or `B`, and each row (x = 0 first)
separated by a comma. For example, a 2x2 board on turn 1 would be given as
`00,00`.

`TURN` is the turn number for the move you are about to make, starting from 1.

Example commands:

| Command | Explanation |
| --- | --- |
| `START;;000,000,000;1;\n` | Request for the opening move of a game using a 3x3 board |
| `SWAP;;0R0,000,000;3;\n` | The opponent has swapped colours on turn 2 and the player agent is about to make move 3 |
| `CHANGE;0,1;0R0,000,000;2;\n` | The opponent changed tile (0,1) to red as their opening move |

Example responses:

| Response | Explanation |
| --- | --- |
| `1,2\n` | Change the tile with the coordinates (1,2) to your colour |
| `-1,-1\n` | Make a swap move |

Important:

- Flush stdout after each response (in C++, use `std::endl` or `fflush`).
- Write any debug text to stderr only. Any other text on stdout breaks the
  protocol.
- Exit when stdin closes. Do not leave processes running after the game.

#### Compiled code and build.sh

Games use the compiled files that you submit. We also rebuild them from your
source, to check that the source is the code that plays. So if your agent
has any compiled part (a program, a shared library, a Python extension
module, or Java `.class` files), you must submit:

1. the source code of every compiled part;
2. the compiled files, at the paths your agent uses them;
3. a script `agents/GroupXYZ/build.sh` that builds every compiled file again
   from the source, at the same paths.

We run `bash agents/GroupXYZ/build.sh` from the repository root, in the build
image (`docker build --target build -t hex-build .`), on a copy of your
folder from which all compiled files have been removed. The build may use the
network (for example to download Rust crates or Go modules). Limits for builds are:
30 minutes, 8 CPUs, 16 GB of memory.

| Outcome | Result |
| --- | --- |
| Our rebuild works | Games use the files you submitted. |
| Your submitted files do not run on the marking machine, but our rebuilt files do | Games use our rebuilt files. |
| Our rebuild fails, or a compiled file has no source | Your agent is marked as an agent that does not run (32%). |
| Your files and our rebuilt files behave very differently | We review your submission by hand. |

The build image has g++ (GCC 13), cmake, ninja, the Python 3.12 headers,
pybind11, Cython, Java 21, Go and Rust (with the `x86_64-unknown-linux-musl`
target). The exact versions are in `/etc/hex-build-tools.txt` and
`/etc/hex-pip-freeze.txt` in the image. If you want to use a different language 
or toolchain, speak to the course staff.

Advice:

- The marking machine is x86-64. Do not compile with `-march=native`; use
  `-march=x86-64-v3` if you want newer CPU instructions. On an Apple Silicon
  Mac, build the images with `--platform linux/amd64`.
- Python extension modules work only with the Python version they were
  compiled for (3.12), so build them in the image.
- C++ code that includes the PyTorch headers (for example
  `torch/extension.h`) must be compiled as C++20 (`-std=c++20`): the headers
  of PyTorch 2.15 do not compile as C++17.
- Go: build with `CGO_ENABLED=0` to get a static program. Rust: build with
  `--target x86_64-unknown-linux-musl` to get a static program.
- Java: `javac --release 21`.
- Test your `build.sh` on a fresh copy of the repository, after you delete
  the compiled files: see step 6 of the
  [submission checklist](SUBMISSION-CHECKLIST.md#6-make-the-tar-file-and-test-it-on-a-fresh-copy).

Example `build.sh` for a C++ program and a Java agent:

```bash
#!/bin/bash
set -e
g++ -O2 -march=x86-64-v3 -o agents/Group007/engine agents/Group007/src/*.cpp
javac --release 21 -d agents/Group007 agents/Group007/MyAgent.java
```

## Running the game

### Basics

#### From Terminal

Run these commands inside the Docker container (see [Docker](#docker)), from
the repository root.

Example commands:

| Command | Explanation |
| --- | --- |
| `python3 Hex.py` | Run the game engine with two default agents |
| `python3 Hex.py -p1 "agents.Group007.MyAgent MyGoodAgent"` | Run the game engine with player 1 as the class `MyGoodAgent` in the file `agents/Group007/MyAgent.py`, and player 2 as default agent |
| `python3 Hex.py -p2 "agents.Group007.MyAgent MyGoodAgent"` | Run the game engine with player 2 as the class `MyGoodAgent` in the file `agents/Group007/MyAgent.py`, and player 1 as default agent |
| `python3 Hex.py -p1 "agents.Group007.MyOtherAgent MyBadAgent" -p2 "agents.Group007.MyAgent MyGoodAgent"` | Run the game engine with player 1 as the class `MyBadAgent` from the file `agents/Group007/MyOtherAgent.py` and player 2 as the class `MyGoodAgent` in the file `agents/Group007/MyAgent.py` |

The first command will play a game between two Naive agents, located at
`agents/DefaultAgents/NaiveAgent.py` with the class name `NaiveAgent`. Each
agent runs in its own process and is paused during the opponent's turn, as in
marked games. By default the game log will also be printed to stderr. The details of the log
formatting are explained in [Log format](#log-format).

To see all the available options and arguments of the engine use
`python3 Hex.py --help`, or see the table below. (Note that quotes must be
used when specifying arguments which contain spaces, i.e. the `-p1` and
`-p2` arguments.)

| Argument | Usage |
| --- | --- |
| `-p1`, `--player1` | Specify the player 1 agent. Format: `agents.GroupXYZ.AgentFile AgentClassName`, e.g. `"agents.Group007.NaiveAgent NaiveAgent"` |
| `-p1Name`, `--player1Name` | Specify the player 1 name (default `Red`) |
| `-p2`, `--player2` | Specify the player 2 agent, in the same format |
| `-p2Name`, `--player2Name` | Specify the player 2 name (default `Blue`) |
| `-v`, `--verbose` | Enable verbose logging |
| `-b`, `--board_size` | Specify the board size\*, e.g. `-b 11` |
| `-l`, `--log` | Save the move history to a log file. If the flag is present, the result is saved to `game.log`; if a filename is provided, to that file. If the flag is not present, the log is printed to stderr. |
| `--seed` | Random seed for the game (player 1 gets the seed, player 2 the seed + 1). Without it, the engine chooses a seed and prints it, so that you can repeat a game. |
| `--debug` | Debug mode: run both agents inside the engine's process, as in the 2025 engine, so that you can use a debugger (`pdb`, `breakpoint()`) in your agent. Agents are not paused, and there are no time limits (time spent in the debugger counts on your clock, but your agent does not lose by `TIMEOUT`). Marked games do not use this. |

\* Your agent will be tested on an 11x11 board; this option is provided for
development.

#### Docker

We run each game in the Docker image built from the `Dockerfile` in the
repository, so test your agent in this image. The commands to build the image
and run the container (for Linux, macOS, Windows PowerShell and Command
Prompt) are in the [README](../README.md). In the container, all the steps
from [From Terminal](#from-terminal) apply. The limits in marked games are in
[Rules in marked games](#rules-in-marked-games).

### Log format

The game log is a CSV file. The following is an example of a log generated
by `python3 Hex.py -l -p1Name Alice -p2Name Bob`:

```csv
1,Alice,RED(x=4, y=5),31517
2,Bob,REDSWAP(),2667
3,Alice,BLUE(x=7, y=7),55590
4,Bob,RED(x=0, y=0),6583
...
Bob,356350
Alice,902392
winner,Alice,WIN
```

Each move line gives the turn number, player name, colour and move, and the
player's **total** time used so far in this game, in nanoseconds. As a
result, line 1 means "at turn 1, player Alice, holding red, played at
(x=4, y=5); Alice has used 31517 ns so far". Line 2 means "Bob played the
swap move (and is now Red)".

The last lines show the total time spent by the player on the red side, then
the blue side, and lastly who the winner is and what the end state is. If a
player crashed or could not be loaded, the lines before the totals start with
`#` and show the error. Lines that start with `# WARNING` show a rule check
(see [Rules in marked games](#rules-in-marked-games)).

As mentioned, by default the engine prints the log to stderr. Using `-l`
writes the log to `game.log` in the directory where you start the engine.
Each time, the file is overwritten. You can save the log to another file by
providing a file path, for example `-l ./test/test.log`.

### cmd.txt

You will specify the path to your agent file using a file named `cmd.txt`,
which will be included within your group folder.

For example, if your agent is located at `agents/Group007/NaiveAgent.py` and
the class name is `VeryNaiveAgent`, then the content of the `cmd.txt` should
be the following:

```text
agents.Group007.NaiveAgent VeryNaiveAgent
```

Note that this is only the location and name of your agent class, and not
any other parts of the command to launch the game engine. It is very
important that `cmd.txt` only includes the file path and the class name, and
not any additional text. No messages, no instructions – this text will be
extracted from the file automatically during the marking process, so if you
add any additional text to the file your code may fail to run.

The string will be the calling argument for performing the agent import, so
the format needs to be precise:

- There should be only one white space between the agent path and the agent
  class name, and only one line.
- Save it as plain text, UTF-8, without a byte order mark (BOM).
- The `GroupXYZ` in the line must match your folder name exactly, including
  the capital G and the three digits.
- If you wrote your agent in an alternative programming language, specify
  the Python agent in `cmd.txt`; you will launch your code from within this
  Python class as explained in [Alternative languages](#alternative-languages).
- Use full import paths in your code, for example
  `from agents.Group007.utils import f`. A plain `from utils import f` fails.
- Open your files with paths built from your file's location, for example
  `os.path.join(os.path.dirname(__file__), "weights.pt")`. Games run from the
  repository root.

## Rules in marked games

- **Time:** each player has 5 minutes in total per game for all its moves (a
  chess clock). Start-up (your `__init__`) must finish within 3 minutes.
  If you go over either limit, you lose that game.
- **Hardware:** your agent gets 8 CPUs (logical CPUs, as with Docker's
  `--cpuset-cpus=0-7`) and 8 GB of memory. There is no GPU available.
- **Pausing:** your agent is paused during the opponent's turn. Its threads
  and child processes stop too, and continue in your next turn. `Hex.py` and
  `check_submission.py` do the same. Do not start processes that leave your
  agent's process group (for example with `setsid` or `start_new_session`).
- **Fresh folder:** each game starts from a fresh copy of your folder. Files
  that your agent writes during a game are not kept.
- **Random seed:** the engine sets a seed for each agent in each game. It
  seeds Python's `random` and NumPy's global generator before your module is
  imported, and PyTorch (`torch.manual_seed`) after your module is imported,
  if your module imports PyTorch. A seed that your agent sets in `__init__`
  or later replaces it. Generators that you make yourself (for example
  `numpy.random.default_rng()`) are not seeded by the engine.
- **Rule checks:** the engine records any process of your agent that uses
  CPU time while your agent is paused, and your agent's peak memory.
  `check_submission.py` shows these warnings. If a marked game shows a rule
  break, we review your games by hand and can deduct marks.
- **No network:** there is no network access during games.
- **Fair play:** do not read or change the game engine, the opponent or the
  timer. We treat this as academic misconduct.
- **Only your folder:** we use our own copy of everything outside
  `agents/GroupXYZ/`, so changes to `src/`, `Hex.py` or other folders are lost.

## How your bot is marked

- Your bot plays 20 bots from last year's class, once as player 1 (Red) and
  once as player 2 (Blue): 40 games.
- **Win mark:** 50 + the number of games your bot wins (maximum 90).
- A bot that is submitted on time but does not run gets 32%. A bot that runs
  but never completes a legal game gets 40%.
- **Speed mark:** from your bot's mean time per move (start-up not included),
  ranked against this year's bots. The fastest bot gets 90%, the median bot
  70% and the slowest 50%.
- The performance mark is (3 × win mark + speed mark) / 4. Details and extra
  credit are in the coursework brief.
