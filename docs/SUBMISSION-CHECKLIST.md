# Submission checklist

Do these steps before you submit, and again after each change to your bot.
Run all commands **inside the Docker container**, from the repository root,
unless a step says something different. The container is the environment of
marked games and is how we guarantee that your bot will run in the same way 
on our marking machine. 

In the examples, the group number is 7, so the folder is `Group007`. Use
your own Canvas group number, always with three digits.

The rules are in [GAME.md](GAME.md). 

**This assessment is AI prohibited:** all submissions must be written by
humans (see the coursework brief).

## 1. Start the container

Follow the [README](../README.md) to build the image and start the container.
Use `--cpuset-cpus=0-7 --memory=16g`, because a marked game has these
resources (leave out `--cpuset-cpus` if your computer has fewer than 8 CPUs).
On an Apple Silicon Mac, build with `--platform linux/amd64`.

IMPORTANT: do not change the `Dockerfile`. If you changed the Dockerfile for 
any reason, build the image again from the original `Dockerfile` on GitHub.
We will use our own copy to build your bot. 

## 2. Check your folder

- [ ] Everything your bot needs is in `agents/Group007/`: code, weights,
      data, and `cmd.txt`. We use our own copy of everything outside this
      folder, so changes to `src/`, `Hex.py` or other folders are lost.
- [ ] `cmd.txt` has one line: `agents.Group007.<module> <ClassName>`, with
      one space. It is plain text, UTF-8, without a byte order mark.
- [ ] Your class inherits from `AgentBase` and calls
      `super().__init__(colour)`.
- [ ] Your imports use full paths, for example
      `from agents.Group007.utils import f`.
- [ ] Your code opens its files with paths built from the file's location,
      for example `os.path.join(os.path.dirname(__file__), "weights.pt")`.
      It does not use absolute paths, and it does not read files outside your
      folder.
- [ ] Your bot uses only the packages in the image (the list is in
      `/etc/hex-pip-freeze.txt` in the container). The folder has no copies
      of libraries.
- [ ] Your bot does not use the network (there is no network in marked
      games). It does not download anything at start-up.
- [ ] Your bot works without a GPU. For PyTorch, load weights with
      `map_location="cpu"`.
- [ ] Your bot does not start processes that leave its process group (for
      example with `setsid` or `start_new_session=True`).
- [ ] An agent in another language reads stdin and writes stdout as
      [GAME.md](GAME.md#message-format) describes, flushes stdout after each
      move, writes debug text to stderr only, and stops when stdin closes.

## 3. If your bot has compiled parts

Compiled parts are programs, shared libraries, Python extension modules and
Java `.class` files.

- [ ] The folder has the source code of every compiled part, the compiled
      files, and `agents/Group007/build.sh`.
- [ ] You compiled the files in the image (Python extension modules must be
      for Python 3.12), without `-march=native`.
- [ ] `build.sh` builds every compiled file again, at the same path. Test
      this in step 6.

## 4. Run check_submission.py

```bash
python3 check_submission.py 7
```

The script checks your folder and `cmd.txt`, loads your bot, and plays three
games as if it were in the marking environment: your bot as player 1, as player 2, and
as player 1 against an opponent that swaps on turn 2. Each bot runs in its
own process and is paused during the opponent's turn. You should check:

- [ ] There is no `FAIL` line. Fix every `FAIL` before you submit.
- [ ] Read every `WARNING` line:
  - start-up above 90 s, or more than 150 s of move time in a game: the limits
    are 3 minutes for start-up and 5 minutes to play the game. The marking machine 
    (Intel Xeon, 2.0 GHz base frequency) can be slower than your computer, so leave a
    margin.
  - `rule check`: a process of your bot used CPU time while your bot was
    paused, or your bot used more than 8 GB of memory. We review games with
    a rule break by hand and can deduct marks.

The script does not tell you how strong your bot is.

## 5. Play more games

Play your bot against itself and against other agents, as both colours. Use
different seeds:

```bash
python3 Hex.py -p1 "agents.Group007.MyAgent MyAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent" --seed 1 -l test1.log
python3 Hex.py -p1 "agents.DefaultAgents.NaiveAgent NaiveAgent" -p2 "agents.Group007.MyAgent MyAgent" --seed 2 -l test2.log
```

Check:
- [ ] Each game ends with `WIN` (see the last line of the log), not
      `TIMEOUT`, `BAD_MOVE`, `CRASH` or `FAILED_LOAD`.
- [ ] The log has no `# WARNING` lines.
- [ ] At the end of each game, `Hex.py` prints the total move time of each
      player. The total of your bot is well below 300 s (5 minutes).
- [ ] Memory: while a game runs, run `docker stats hex` in a second terminal
      on your computer (not in the container). The container's memory must
      stay well below 16 GB (8 GB for each bot).

## 6. Make the .tar file and test it on a fresh copy

Make the file inside the container, from the `agents` folder:

```bash
cd agents
tar -cvf Group007.tar Group007
tar -tf Group007.tar | head      # file names
tar -tvf Group007.tar | head     # file names with permissions
cd ..
```

Check:
- [ ] The file name is `Group007.tar`, and every line of `tar -tf` starts
      with `Group007/`.
- [ ] The file is at most 1 GB (`ls -lh agents/Group007.tar`).
- [ ] Programs that must run have `x` in their permissions in the
      `tar -tvf` output (for example `-rwxr-xr-x`).

Then test the `.tar` file on a fresh copy of the repository, so that you
find files that your bot needs but that are not in your folder. If your bot
has compiled parts, use the build image (`hex-build`), and also delete the
compiled files and build them again with your `build.sh`:

```bash
# on your computer, in the repository folder (bash; for PowerShell use ${PWD},
# for Command Prompt use %cd%, as in the README)
docker run -it --rm -v "$(pwd)":/home/hex hex-build bash

# in the container
cd /tmp
git clone https://github.com/sara-es/COMP34111-AI-Games-Hex.git fresh
tar -xf /home/hex/agents/Group007.tar -C fresh/agents
cd fresh
python3 check_submission.py 7
```

If your bot has compiled parts, then also delete the compiled files in
`fresh/agents/Group007/`, and do this:

```bash
bash agents/Group007/build.sh
python3 check_submission.py 7
```

If your bot has no compiled parts, you can use the image `hex` instead of
`hex-build`.

Check:
- [ ] `check_submission.py` passes on the fresh copy.
- [ ] If your bot has compiled parts: `build.sh` works on the fresh copy, and
      `check_submission.py` passes with the rebuilt files.

## 7. Submit

- [ ] Submit `Group007.tar` to the bot assignment on Canvas, one submission
      per group. Dates and deadlines are on Canvas.
- [ ] To use the dry run, submit before the dry-run deadline in
      [GAME.md](GAME.md#submission). We tell you if your bot runs
      within the time limit and if we can rebuild its compiled parts.
- [ ] After each change to your bot, do this checklist again and submit
      again.
