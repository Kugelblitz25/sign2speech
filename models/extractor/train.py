from collections import namedtuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from models.extractor.model import ModifiedI3D, ModifiedR2P1D, ModifiedX3D
from utils import Config, EarlyStopping, create_path, save_model

models = {
    "i3d": ModifiedI3D,
    "x3d": ModifiedX3D,
    "r2p1d": ModifiedR2P1D,
}


class Trainer:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        video_root: str,
        num_classes: int,
        model_name: str,
        train_config: namedtuple,
        checkpoint_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_config = train_config
        self.checkpoint_path = create_path(checkpoint_path)

        print(f"Using Device: {self.device}")

        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)

        print(f"Num Classes: {num_classes}")

        self.model = self.load_model(model_name, num_classes).to(self.device)
        self.train_loader = self.get_dataloader(train_data, video_root)
        self.val_loader = self.get_dataloader(val_data, video_root)

    def load_model(self, model_name, num_classes):
        model = models.get(model_name, None)
        if model is None:
            raise ValueError("Invalid Model Name")
        model = model(num_classes).to(self.device)
        for i in range(self.train_config.freeze):
            for param in model.backbone.blocks[i].parameters():
                param.requires_grad = False
        return model

    def get_dataloader(self, data: pd.DataFrame, video_root: str):
        dataset = WLASLDataset(data, video_root)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
        )
        return dataloader

    def train_epoch(self, epoch: int):
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.train_config.epochs}"
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
            lr=float(self.train_config.lr),
            weight_decay=float(self.train_config.weight_decay),
        )
        early_stopping = EarlyStopping(
            patience=self.train_config.patience, verbose=True
        )
        # self.optimizer = optim.Adam(
        # self.model.parameters(),
        # lr=self.train_config.lr,
        # weight_decay=self.train_config.weight_decay,
        # )
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.train_config.lr,
            momentum=0.9,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.train_config.scheduler_factor,
            patience=self.train_config.scheduler_factor,
        )
        early_stopping = EarlyStopping(
            patience=self.train_config.patience, verbose=True
        )

        # Training
        best_train_acc = 0.0
        best_test_acc = 0.0

        for epoch in range(self.train_config.epochs):
            train_acc, train_loss = self.train_epoch(epoch)
            val_acc, val_loss = self.validate()
            scheduler.step(val_loss, epoch)
            print(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%"
            )

            # Check early stopping condition
            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                print("Best Model. Saving ...")
                save_model(
                    self.model,
                    self.train_config._asdict(),
                    val_loss,
                    self.checkpoint_path / f"checkpoint_best_{self.model.name}.pt",
                )
                best_train_acc, best_test_acc = train_acc, val_acc

            if early_stopping.early_stop and self.train_config.enable_earlystop:
                print("Early stopping triggered. Stopping training.")
                save_model(
                    self.model,
                    self.train_config._asdict(),
                    val_loss,
                    self.checkpoint_path / f"checkpoint_final_{self.model.name}.pt",
                )
                break

        save_model(
            self.model,
            self.train_config._asdict(),
            val_loss,
            self.checkpoint_path / f"checkpoint_final_{self.model.name}.pt",
        )

        return best_train_acc, best_test_acc, train_acc, val_acc


if __name__ == "__main__":
    config = Config("Train and validate video classification model")

    trainer = Trainer(
        config.data.processed.csvs.train,
        config.data.processed.csvs.val,
        config.data.processed.videos,
        config.n_words,
        config.extractor.model,
        config.extractor.training,
        config.extractor.checkpoints,
    )
    trainer.train()
