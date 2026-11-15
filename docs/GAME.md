# Hex game engine: documentation

COMP34111 AI and Games. This page tells you how the game engine works, how to
write your bot, and how we run and mark it. For Docker commands, see the
[README](../README.md). Dates and deadlines are on Canvas.

- [Submission checklist](#submission-checklist)
- [The game](#the-game)
- [Write a bot in Python](#write-a-bot-in-python)
- [Write a bot in another language](#write-a-bot-in-another-language)
- [Run games](#run-games)
- [Game log](#game-log)
- [cmd.txt](#cmdtxt)
- [Rules in marked games](#rules-in-marked-games)
- [How your bot is marked](#how-your-bot-is-marked)

## Submission checklist

Submit one zip file of your group folder, `agents/GroupN/`, where N is your
Canvas group number. When we unzip it, the top level must be the folder
`GroupN/`. The maximum size is 1 GB.

1. A correct `cmd.txt` in `agents/GroupN/` (see [cmd.txt](#cmdtxt)).
2. A Python class that inherits from `AgentBase`: your own bot, or a class
   that starts your program in another language (see
   [Write a bot in another language](#write-a-bot-in-another-language)).
3. All other files that your bot needs: Python modules, weights, the source
   code and the compiled files of a program in another language. We do not
   compile your code. Compile it inside the Docker container, and include both
   the source and the compiled files.
4. Do not put copies of libraries in your folder. Use only the packages in
   the Docker image.

Before you submit, run this inside the container, from the repository root:

```bash
python3 check_submission.py N
```

It checks your folder and `cmd.txt` as the marking scripts do. Then it plays
three games with the real time limits: your bot as player 1, your bot as
player 2, and your bot as player 1 against an opponent that swaps on turn 2.
We do not accept fixes after the deadline, so submit early for the dry run if
you want us to check that your bot runs on our machine.

## The game

- The rules of Hex: [Hex (board game)](https://en.wikipedia.org/wiki/Hex_(board_game)).
- The board has 11 × 11 tiles. A tile has the coordinates `(x, y)`:
  `x` is the **row** (0 = top) and `y` is the **column** (0 = left). In code,
  the tile is `board.tiles[x][y]`, and a move to it is `Move(x, y)`.
- Each row is drawn half a tile to the right of the row above it. So the
  tile `(x, y)` has six neighbours: `(x, y-1)` and `(x, y+1)` in the same row,
  `(x-1, y)` and `(x-1, y+1)` in the row above, and `(x+1, y-1)` and
  `(x+1, y)` in the row below. Example: the neighbours `N` of the tile
  `X = (1, 2)`:

  ```text
  y:      0   1   2   3   4
  x=0      .   .   N   N   .
  x=1        .   N   X   N   .
  x=2          .   N   N   .   .
  ```

- Red connects the top row (x = 0) to the bottom row (x = 10). Blue connects
  the left column (y = 0) to the right column (y = 10).
- Player 1 is Red and moves first (turn 1). Player 2 is Blue.
- Swap (pie rule): on turn 2 only, player 2 can return `Move(-1, -1)` instead
  of a normal move. Then the two players change colours: player 2 becomes
  Red and owns the stone that is already on the board, and player 1 becomes
  Blue and plays turn 3. The engine changes the `colour` property of both
  agents. A swap on any other turn is an illegal move.
- A game ends in one of these ways:

  | End | Meaning | Winner |
  | --- | --- | --- |
  | `WIN` | A player connects its two sides of the board. | That player |
  | `BAD_MOVE` | A player plays on a tile that is not empty, plays outside the board, swaps when it is not turn 2, or returns something that is not a `Move`. | The opponent |
  | `TIMEOUT` | A player uses more than its total time (see [Rules in marked games](#rules-in-marked-games)). | The opponent |
  | `CRASH` | A player's `make_move` raises an exception. | The opponent |

## Write a bot in Python

Your bot is a class that inherits from `AgentBase` (in `src/AgentBase.py`).
`agents/DefaultAgents/NaiveAgent.py` is a simple example: it plays a random
empty tile, and swaps on turn 2 half of the time.

- `__init__(self, colour)`: the engine calls this once when the game starts,
  with your colour (`Colour.RED` or `Colour.BLUE`). Do your set-up here, for
  example load weights. Call `super().__init__(colour)`.
- `make_move(self, turn, board, opp_move)`: the engine calls this on each of
  your turns. It must return a `Move` (from `src/Move.py`).
  - `turn`: the turn number of the move you are about to make, from 1.
  - `board`: a copy of the current `Board`. `board.tiles` is a list of rows;
    each `Tile` has the properties `x`, `y` and `colour` (`None` if empty).
  - `opp_move`: the opponent's last move. It is `None` on turn 1, and
    `Move(-1, -1)` if the opponent swapped. After a swap, your `self.colour`
    is already changed.
- To play a tile, return `Move(x, y)`. To swap, return `Move(-1, -1)`.

## Write a bot in another language

`agents/DefaultAgents/ExternalAgent.py` is a Python agent class that starts a
program as a separate process and talks to it through stdin and stdout.
`agents/DefaultAgents/NaiveAgent.java` is an example program in Java. You can
use and change this code as you like; the interface is not part of the
assessment.

To start your own program, change the command in `ExternalAgent.__init__`:

```python
self.agent_process = Popen(
    ["java", "-cp", "agents/DefaultAgents", "NaiveAgent", colour.to_protocol_char(), "11"],
    stdout=PIPE, stdin=PIPE, text=True,
)
```

This is the same as the terminal command
`java -cp agents/DefaultAgents NaiveAgent R 11` for a Red agent. Copy the
class into your group folder and name it in your `cmd.txt`.

### Message format

On each of your turns, the engine sends one line to your program's stdin:

```text
COMMAND;MOVE;BOARD;TURN;
```

Your program answers with one line on stdout: `x,y`, or `-1,-1` to swap.
Every line ends with a newline (`\n`).

- `COMMAND` is one of:
  - `START`: you make the first move of the game (you are Red, turn 1).
  - `CHANGE`: the opponent played a tile. `MOVE` gives its coordinates as
    `x,y`.
  - `SWAP`: the opponent swapped on turn 2. You are now Blue.
- `MOVE` is empty for `START` and `SWAP`.
- `BOARD` is the board at the start of your turn: one string per row (x = 0
  first), separated by commas, with `0`, `R` or `B` for each tile (y = 0
  first). A 2 × 2 empty board is `00,00`.
- `TURN` is the turn number of the move you are about to make, from 1.

If your program swaps on turn 2, it is Red from then on. The engine does not
send a message about your own swap.

Examples on a 3 × 3 board:

| Message | Meaning |
| --- | --- |
| `START;;000,000,000;1;` | Make the first move of the game. |
| `CHANGE;0,1;0R0,000,000;2;` | The opponent (Red) played tile (0, 1) on turn 1. You make move 2. |
| `SWAP;;0R0,000,000;3;` | The opponent swapped on turn 2. You are now Blue and make move 3. |

| Answer | Meaning |
| --- | --- |
| `1,2` | Play tile (1, 2). |
| `-1,-1` | Swap (turn 2 only). |

Important:

- Flush stdout after each answer (in C++, use `std::endl` or `fflush`).
- Write debug text to stderr only. Any other text on stdout breaks the
  protocol.
- Exit when stdin closes. Do not leave processes running after the game.
- Compile inside the Docker container and include the compiled files.

## Run games

Run these commands inside the Docker container, from the repository root.

| Command | What it does |
| --- | --- |
| `python3 Hex.py` | Two `NaiveAgent` bots play each other. |
| `python3 Hex.py -p1 "agents.Group0.MyAgent MyGoodAgent"` | Player 1 is the class `MyGoodAgent` in `agents/Group0/MyAgent.py`; player 2 is `NaiveAgent`. |
| `python3 Hex.py -p2 "agents.Group0.MyAgent MyGoodAgent"` | The same, with your bot as player 2. |
| `python3 Hex.py -p1 "agents.Group0.MyOtherAgent MyBadAgent" -p2 "agents.Group0.MyAgent MyGoodAgent"` | Two of your own bots play each other. |

Put quotes around the values of `-p1` and `-p2`, because they contain a space.
All options (`python3 Hex.py --help`):

| Option | Use |
| --- | --- |
| `-p1`, `--player1` | Player 1 agent, as `agents.GroupX.AgentFile AgentClassName` |
| `-p1Name`, `--player1Name` | Player 1 name (default `Red`) |
| `-p2`, `--player2` | Player 2 agent, as above |
| `-p2Name`, `--player2Name` | Player 2 name (default `Blue`) |
| `-v`, `--verbose` | More detailed output |
| `-b`, `--board_size` | Board size, for example `-b 7`. Marked games use 11. |
| `-l`, `--log` | Write the game log to `game.log`, or to the file you give, for example `-l test/test.log`. The file is overwritten. Without `-l`, the log goes to stderr. |

## Game log

The log is a CSV file. This example is from `python3 Hex.py -l` with the
player names Alice and Bob:

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

- Each move line gives: the turn, the player's name, the player's colour and
  move, and the player's **total** time used so far in this game, in
  nanoseconds. The colour is the colour after the move: in line 2, Bob
  swapped and is now Red.
- After the moves: the total time of each player (Red first), then the
  winner and the end state.
- If a player crashed, the lines before the totals start with `#` and show
  the error.

## cmd.txt

`cmd.txt` tells the marking scripts which class to load. If your bot is the
class `VeryNaiveAgent` in `agents/Group7/NaiveAgent.py`, then `cmd.txt` is:

```text
agents.Group7.NaiveAgent VeryNaiveAgent
```

- One line only: the module path, exactly one space, and the class name. No
  other text.
- Save it as plain text, UTF-8, without a byte order mark (BOM).
- The `GroupN` in the line must match your folder name exactly, including
  the capital G.
- For a bot in another language, give the Python class that starts your
  program.
- Use full import paths in your code, for example
  `from agents.Group7.utils import f`. A plain `from utils import f` fails.
- Open your files with paths built from your file's location, for example
  `os.path.join(os.path.dirname(__file__), "weights.pt")`. Games run from the
  repository root.

## Rules in marked games

- **Time:** each player has 5 minutes in total per game for all its moves (a
  chess clock). Start-up (your `__init__`) must finish within 3 minutes.
  If you go over either limit, you lose that game.
- **Hardware:** your bot gets 8 CPU cores and 8 GB of memory. There is no GPU;
  PyTorch is installed but uses the CPU.
- **Pausing:** your bot is paused during the opponent's turn. Its threads and
  child processes stop too, and continue in your next turn. (`Hex.py` and
  `check_submission.py` do not pause bots yet.)
- **Fresh folder:** each game starts from a fresh copy of your folder. Files
  that your bot writes during a game are not kept.
- **Random seed:** your bot can set its own random seed. If it does not, the
  engine sets one for each game.
- **No network:** there is no network during games.
- **Fair play:** do not read or change the game engine, the opponent or the
  timer. We treat this as academic misconduct.
- **Only your folder:** we use our own copy of everything outside
  `agents/GroupN/`, so changes to `src/`, `Hex.py` or other folders are lost.

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
