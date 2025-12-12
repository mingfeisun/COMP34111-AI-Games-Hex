from network_dev.models.dummy import DummyModel
import torch
from pathlib import Path

def train(model):
    pass  # replace with actual training

def create_train_and_save():
    model = DummyModel()
    train(model)

    save_path = Path(__file__).resolve().parents[2] / "saved_models/dummy_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model, save_path)
    return model

if __name__ == "__main__":
    create_train_and_save()
