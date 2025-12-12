import sys
from pathlib import Path
import torch

# Add the project root to sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Now you can safely import your model
from network_dev.models.dummy import DummyModel
from network_dev.models.hex_neural_net import HexNeuralNet

def train(model):
    # TODO: implement training logic
    pass


def create_train_and_save():
    model = HexNeuralNet()
    train(model)

    save_dir = PROJECT_ROOT / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "hex_neural_net.pth"
    torch.save(model, str(save_path))

    print(f"Model saved at {save_path}")
    return model

if __name__ == "__main__":
    create_train_and_save()
