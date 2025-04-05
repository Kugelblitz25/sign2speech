import logging
from pathlib import Path

import torch

from utils.common import create_path


class EarlyStopping:
    def __init__(
        self, patience: int = 10, verbose: bool = False, delta: float = 0
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                logging.warning(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_model(
    model: torch.nn.Module, config: dict[str, float], loss: float, path: str | Path
) -> None:
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "loss": loss,
    }
    torch.save(save_data, create_path(path))


def load_model_weights(
    model: torch.nn.Module, path: str | Path, device: torch.device
) -> None:
    logging.info(f"Weights Loaded from {path}")
    weights = torch.load(path, weights_only=False, map_location=device)[
        "model_state_dict"
    ]
    model.load_state_dict(weights)
