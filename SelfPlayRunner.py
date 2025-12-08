# generate_selfplay_data.py

import pickle
from pyexpat import model
from agents.Group21.SelfPlay import SelfPlay
from src.Game import Game
import numpy as np
from network_dev.saved_models.RandomNN import RandomNN
    
def main():
    # Load a model from saved models (here we use a random model as a placeholder)
    nn = RandomNN()

    # Initialize SelfPlay engine
    
    self_play_engine = SelfPlay(
        neural_net=nn,
        game_cls=Game,  
        simulations=50
    )

    # Generate self-play training data
    num_games = 10  # number of games to generate
    training_examples = []
    for i in range(num_games):
        print(f"Playing game {i+1}/{num_games}...")
        examples = self_play_engine.play_game()
        training_examples += examples

    # Save training examples
    save_file = "training_data.pkl"
    with open(save_file, "wb") as f:
        pickle.dump(training_examples, f)

    print(f"Saved {len(training_examples)} training samples to {save_file}")



if __name__ == "__main__":
    main()
