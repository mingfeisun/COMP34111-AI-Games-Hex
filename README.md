# COMP34111 AI and Games: Hex

This repository holds the Hex game engine for the COMP34111 group project. We
mark every bot inside the Docker image built from this `Dockerfile`, so make
sure your bot works inside the container.

## Build the image

```bash
docker build -t hex .
```

The build takes a while. On Linux or macOS you can add
`--build-arg UID=$(id -u)`, so that files you create in the container belong
to your user.

## Run the container

Linux, macOS or WSL (bash):

```bash
docker run -it --cpus=8 --memory=8g --name hex -v "$(pwd)":/home/hex hex bash
```

Windows PowerShell:

```powershell
docker run -it --cpus=8 --memory=8g --name hex -v "${PWD}:/home/hex" hex bash
```

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
python3 Hex.py -p1 "agents.Group7.MyAgent MyAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent"
```

`python3 Hex.py --help` shows all options.

## Check your submission

```bash
python3 check_submission.py 7      # checks agents/Group7/
```

## Limits in marked games

- 5 minutes per player per game for all moves (chess clock).
- 3 minutes for your agent's start-up (`__init__`).
- 8 CPU cores and 8 GB of memory per agent. There is no GPU. PyTorch is
  installed, but it uses the CPU.
- No network during games. Use only the packages in the image.

The full rules are in the coursework instructions on Canvas.

## Bots in other languages

`agents/DefaultAgents/ExternalAgent.py` shows how a Python agent class starts a
program in another language and talks to it. The image has a Java JDK and
`g++`. Compile inside the container, for example:

```bash
javac -d agents/DefaultAgents agents/DefaultAgents/NaiveAgent.java
```

## Run the tests

```bash
python3 -m unittest discover
```

## Game documentation

How the engine works, the message format for other languages, the log
format, `cmd.txt` and the rules for marked games: [docs/GAME.md](docs/GAME.md).
