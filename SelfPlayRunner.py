# generate_selfplay_data.py
import torch
import pickle
from pyexpat import model
from agents.Group21.SelfPlay import SelfPlay
from src.Game import Game
import numpy as np
from pathlib import Path
from network_dev.saved_models.RandomNN import RandomNN
from network_dev.models.dummy import DummyModel
    
def load_model():
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "saved_models"
    model_path = models_dir / "dummy_model.pth"
    if not model_path.exists():
        # fallback to legacy location
        alt_path = project_root / "network_dev" / "saved_models" / "dummy_model.pth"
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model not found: {model_path} (tried fallback {alt_path})")

    # Robust loading: try loading as state_dict (weights_only) first; if that fails, allowlist DummyModel and load full object.
    model_path_str = str(model_path)
    nn = None
    # 1) Try weights-only load (state_dict)
    try:
        state = torch.load(model_path_str, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            nn = DummyModel()
            nn.load_state_dict(state)
    except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    # 2) If state_dict path didn't work, try loading the full object with safe_globals/add_safe_globals
    if nn is None:
        try:
            # Prefer the context manager if available
            if hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([DummyModel]):
                    nn = torch.load(model_path_str, map_location="cpu", weights_only=False)
            else:
                # Fallback to add_safe_globals then load
                if hasattr(torch.serialization, "add_safe_globals"):
                    torch.serialization.add_safe_globals([DummyModel])
                nn = torch.load(model_path_str, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e
    
    return nn

def main():
    
    # Load model 
    nn = load_model()

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
