# Group 25 Monte Carlo Tree Search Hex Agent

## Introduction
1. Create a virtual environment using:
`python -m venv venv` / `python3 -m venv venv`
2. Install the requirements in the base of the repository using:
`pip install -r requirements.txt` this installs the packages for the game and some used for development.

## Testing
1. If all requirements are installed, you can run the tests using:
`pytest agents/Group25/tests/`

## Profiling
1. Run the game with the profiler using: `python -m cProfile -o profile.out Hex.py -p1 "agents.Group25.MCTSAgent MCTSAgent"`
2. This will output a file called `profile.out` which can be read using `snakeviz profile.out`. This will open a web browser containing details of the profiling.
