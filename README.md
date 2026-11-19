# COMP34111 AI and Games: Hex

This repository holds the Hex game engine for the COMP34111 group project. We
play and mark every bot inside the Docker image built from this `Dockerfile`,
so make sure your bot works inside the container.

How the engine works, the rules for marked games and how bots are marked:
[docs/GAME.md](docs/GAME.md).

## The environment

- Ubuntu 24.04, x86-64 (also the architecture of the marking machine).
- Python 3.12 with PyTorch 2.15 (CPU only), NumPy, pandas, SciPy and
  scikit-learn. The exact versions are in `/etc/hex-pip-freeze.txt` in the
  image.
- Java 21 (JDK) and g++.
- There is **no GPU** available in marked games.

## Build the image

```bash
docker build -t hex .
```

The build takes a while. **On an Apple Silicon Mac**, build an x86-64 image, or
your compiled code will not run on the marking machine:

```bash
docker build --platform linux/amd64 -t hex .
```

If your bot has compiled parts (C, C++, Go, Rust, Java, Python extension
modules), also build the build image, which adds the compilers and tools we
use to rebuild your code (see docs/GAME.md):

```bash
docker build --target build -t hex-build .
```

If you would like to use a different language or toolchain, speak to the course staff. 
We will try to help, but cannot guarantee we are able to accommodate all requests. 
See [GAME.md](docs/GAME.md) for more detail.

## Run the container

Linux, macOS or WSL (bash):

```bash
docker run -it --cpuset-cpus=0-7 --memory=16g --name hex -v "$(pwd)":/home/hex hex bash
```

Windows PowerShell:

```powershell
docker run -it --cpuset-cpus=0-7 --memory=16g --name hex -v "${PWD}:/home/hex" hex bash
```

Windows Command Prompt (cmd.exe):

```bat
docker run -it --cpuset-cpus=0-7 --memory=16g --name hex -v "%cd%:/home/hex" hex bash
```

`--cpuset-cpus=0-7` gives the container 8 virtual CPU cores. This is how many your bot will get during marking. 
If your computer has fewer than 8 CPUs, leave the option out. 16 GB is for both bots of a game together (8 GB
each).

The container starts in `/home/hex`, which is this repository. Changes there
are also changes on your computer. Type `exit` to stop the container. To start
it again, use `docker start -i hex`. To remove it, use `docker rm hex`.

## Play a game

```bash
python3 Hex.py
```

By default, two `agents/DefaultAgents/NaiveAgent.py` agents play each other.
To choose the agents, give the module and class name of each player:

```bash
python3 Hex.py -p1 "agents.Group007.MyAgent MyAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent"
```

`python3 Hex.py --help` shows all options.

## Check your submission

```bash
python3 check_submission.py Group007      # checks agents/Group007/
```

## Run the tests

```bash
python3 -m unittest discover
```
