import subprocess
import argparse
import time

def extract_winner(output: str):
    for line in output.splitlines():
        if line.startswith("winner,"):
            return line.split(",")[1]
    return None


def run_games(num_games, player1, player1Name, player2, player2Name):
    wins = 0
    total = 0
    game_times = []

    for i in range(num_games):
        print(f"Running game {i+1}/{num_games}...")

        start_time = time.time()

        result = subprocess.run(
            [
                "python3", "Hex.py",
                "-p1", player1,
                "-p1Name", player1Name,
                "-p2", player2,
                "-p2Name", player2Name
            ],
            capture_output=True,
            text=True
        )

        end_time = time.time()
        duration = end_time - start_time
        game_times.append(duration)

        print(f"Game {i+1} completed in {duration:.3f} seconds")

        if result.returncode != 0:
            print("Game crashed:")
            print(result.stderr)
            continue

        output = result.stdout + result.stderr
        winner = extract_winner(output)

        if winner == player1Name:
            wins += 1

        total += 1

    return wins, total, game_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_games", type=int, default=10)
    parser.add_argument("-p1", "--player1", type=str, default="agents.Group21.MCTSAgent MCTSAgent")
    parser.add_argument("-p1Name", "--player1Name", type=str, default="My Agent")
    parser.add_argument("-p2", "--player2", type=str, default="agents.Group21.RandomAgent RandomAgent")
    parser.add_argument("-p2Name", "--player2Name", type=str, default="Random Agent")

    args = parser.parse_args()

    wins, total, game_times = run_games(args.num_games, args.player1, args.player1Name, args.player2, args.player2Name)

    print("\n=== RESULTS ===")
    print(f"Total Games: {total}")
    print(f"Wins for {args.player1}: {wins}")
    print(f"Win Rate: {wins/total*100:.2f}%")

    print("\n=== TIME STATISTICS ===")
    print(f"Average Game Time: {sum(game_times)/len(game_times):.3f} s")
    print(f"Fastest Game: {min(game_times):.3f} s")
    print(f"Slowest Game: {max(game_times):.3f} s")
