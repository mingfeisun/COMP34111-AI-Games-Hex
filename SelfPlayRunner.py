# generate_selfplay_data.py
import torch
import pickle
from pyexpat import model
from agents.Group21.SelfPlay import SelfPlay
from src.Game import Game
import numpy as np
from pathlib import Path
from saved_models.RandomNN import RandomNN
from network_dev.models.dummy import DummyModel
    
def load_model(model_name="hex_neural_net.pth"):
    # Project root (two levels up from this script)
    project_root = Path(__file__).resolve().parents[0]
    models_dir = project_root / "saved_models"
    model_path = models_dir / model_name

    # Fallback to legacy location
    if not model_path.exists():
        alt_path = project_root / "network_dev" / "saved_models" / model_name
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model not found: {model_path} (tried fallback {alt_path})")

    # Load the full saved model
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    return model

def main():
    dummy = "dummy_model.pth"
    alpha = "hex_neural_net.pth"
    # Load model 
    nn = load_model(model_name=alpha)

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
