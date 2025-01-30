import argparse
from collections import namedtuple
from pathlib import Path

import torch
import yaml


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


class Config:
    def __init__(self, desc: str):
        parser = argparse.ArgumentParser(description=desc)

        parser.add_argument(
            "--config_file",
            type=str,
            default="config.yaml",
            help="Path to the config file",
        )
        args = parser.parse_args()

        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)

        self.config = self.parse_dict("RootCFG", config)

    def parse_dict(self, name: str, dictionary: dict):
        nt = namedtuple(name, dictionary.keys())
        processed_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                processed_dict[key] = self.parse_dict(key, value)
            elif isinstance(value, list):
                processed_dict[key] = [
                    self.parse_dict(f"item{i}", item)
                    if isinstance(item, dict)
                    else item
                    for i, item in enumerate(value)
                ]
            else:
                processed_dict[key] = value

        return nt(**processed_dict)

    def __getattr__(self, name: str):
        return getattr(self.config, name)
