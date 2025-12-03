import subprocess
import argparse


def extract_winner(output: str):
    for line in output.splitlines():
        if line.startswith("winner,"):
            return line.split(",")[1]
    return None


def run_games(num_games, target_player):
    wins = 0
    total = 0

    for i in range(num_games):
        print(f"Running game {i+1}/{num_games}...")

        result = subprocess.run(
            [
                "python3", "Hex.py",
                "-p1", "agents.Group21.MCTSAgent MCTSAgent",
                "-p1Name", target_player,
                "-p2", "agents.Group21.RandomAgent RandomAgent",
                "-p2Name", "RANDOM"
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("Game crashed:")
            print(result.stderr)
            continue

        output = result.stdout + result.stderr
        winner = extract_winner(output)

        if winner == target_player:
            wins += 1

        total += 1

    return wins, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_games", type=int, default=10)
    parser.add_argument("-p", "--player", type=str, required=True)
    args = parser.parse_args()

    wins, total = run_games(args.num_games, args.player)

    print("\n=== RESULTS ===")
    print(f"Total Games: {total}")
    print(f"Wins for {args.player}: {wins}")
    print(f"Win Rate: {wins/total*100:.2f}%")
