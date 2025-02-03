import torch
from pathlib import Path


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_model(model, config: dict, loss: float, path: str):
    path = create_path(path)
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "loss": loss,
    }
    torch.save(save_data, path)


def load_model_weights(model, path: str, device: torch.device):
    print(f"Weights Loaded from {path}")
    weights = torch.load(path, weights_only=True, map_location=device)[
        "model_state_dict"
    ]
    model.load_state_dict(weights)
    return model


def create_path(path_str: str):
    path = Path(path_str)
    if not path.suffix:
        path.mkdir(exist_ok=True, parents=True)
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
    return path
