import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import EarlyStopping, save_model, create_path, load_config

from models.transformer.dataset import SpectrogramDataset
from models.transformer.model import SpectrogramGenerator


class Trainer:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        specs_csv: str,
        train_config: dict,
        checkpoint_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_config = train_config
        self.checkpoint_path = create_path(checkpoint_path)
        print(f"Using Device: {self.device}")

        self.model = SpectrogramGenerator().to(self.device)
        self.train_loader = self.get_dataloader(train_data_path, specs_csv)
        self.val_loader = self.get_dataloader(val_data_path, specs_csv)

    def get_dataloader(self, features_csv, spec_csv):
        dataset = SpectrogramDataset(features_csv, spec_csv)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=self.train_config["num_workers"],
        )
        return dataloader

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for features, spectrograms in tqdm(
            self.train_loader, f"Epoch {epoch}/{self.train_config['epochs']}"
        ):
            features = features.to(self.device)
            spectrograms = spectrograms.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, spectrograms)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for features, spectrograms in tqdm(self.val_loader, "Validation"):
                features = features.to(self.device)
                spectrograms = spectrograms.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, spectrograms)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config["lr"])
        early_stopping = EarlyStopping(
            patience=self.train_config["patience"], verbose=True
        )

        for epoch in range(self.train_config["epochs"]):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                print("Best Model. Saving ...")
                save_model(
                    self.model,
                    self.train_config,
                    val_loss,
                    self.checkpoint_path / "checkpoint_best.pt",
                )

            if early_stopping.early_stop and self.train_config["enable_earlystop"]:
                print("Early stopping triggered. Stopping training.")
                save_model(
                    self.model,
                    self.train_config,
                    val_loss,
                    self.checkpoint_path / "checkpoint_final.pt",
                )
                break

        save_model(
            self.model,
            self.train_config,
            val_loss,
            self.checkpoint_path / "checkpoint_final.pt",
        )


if __name__ == "__main__":
    config = load_config("Transforming video features into spectrogram features")
    train_data_path = config["data"]["processed"]["vid_features_train"]
    val_data_path = config["data"]["processed"]["vid_features_test"]
    specs_csv = config["data"]["processed"]["specs"]
    checkpoint_path = config["transformer"]["checkpoints"]
    train_config = config["transformer"]["training"]

    trainer = Trainer(
        train_data_path, val_data_path, specs_csv, train_config, checkpoint_path
    )
    trainer.train()
