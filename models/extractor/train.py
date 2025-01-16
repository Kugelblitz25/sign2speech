import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset, video_transform
from models.extractor.model import ModifiedI3D
from utils import EarlyStopping, save_model, load_config, create_path


class Trainer:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        video_root: str,
        num_classes: int,
        train_config: dict,
        checkpoint_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_config = train_config
        self.checkpoint_path = create_path(checkpoint_path)
        print(f"Using Device: {self.device}")

        with open(train_data_path) as f:
            train_data = json.load(f)

        with open(val_data_path) as f:
            val_data = json.load(f)

        print(f"Num Classes: {num_classes}")

        self.model = ModifiedI3D(num_classes).to(self.device)
        self.train_loader = self.get_dataloader(train_data, video_root)
        self.val_loader = self.get_dataloader(val_data, video_root)

    def get_dataloader(self, data: list[dict], video_root: str):
        dataset = WLASLDataset(data, video_root, transform=video_transform())
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=self.train_config["num_workers"],
        )
        return dataloader

    def train_epoch(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.train_config['epochs']}"
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            _, outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(self.train_loader)
        train_acc = 100.0 * correct_train / total_train
        return train_acc, train_loss

    def validate(self):
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc=f"Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct_val / total_val
        return val_acc, val_loss

    def train(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
        )
        early_stopping = EarlyStopping(
            patience=self.train_config["patience"], verbose=True
        )

        # Training
        for epoch in range(self.train_config["epochs"]):
            train_acc, train_loss = self.train_epoch(epoch)
            val_acc, val_loss = self.validate()
            print(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%"
            )

            # Check early stopping condition
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
    config = load_config("Train and validate video classification model")

    train_data_path = config["data"]["processed"]["train_data"]
    val_data_path = config["data"]["processed"]["test_data"]
    video_root = config["data"]["processed"]["videos"]
    num_classes = config["n_words"]
    train_config = config["extractor"]["training"]
    checkpoint_path = config["extractor"]["checkpoints"]

    trainer = Trainer(
        train_data_path,
        val_data_path,
        video_root,
        num_classes,
        train_config,
        checkpoint_path,
    )
    trainer.train()
