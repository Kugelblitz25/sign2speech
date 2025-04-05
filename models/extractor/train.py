from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.extractor.dataset import WLASLDataset
from models.extractor.model import Extractor
from utils.common import create_path, get_logger
from utils.config import ExtractorTraining as TrainConfig
from utils.config import load_config
from utils.model import EarlyStopping, save_model

logger = get_logger("logs/extractor_training.log")


class Trainer:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        video_root: str,
        model: Extractor,
        train_config: TrainConfig,
        checkpoint_path: str,
    ) -> None:
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.train_config = train_config
        self.checkpoint_path = create_path(checkpoint_path)

        logger.debug(f"Using Device: {self.device}")

        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)

        self.model = model.to(self.device)
        self.train_loader = self.get_dataloader(train_data, video_root)
        self.val_loader = self.get_dataloader(val_data, video_root)

    def get_dataloader(self, data: pd.DataFrame, video_root: str) -> DataLoader:
        dataset = WLASLDataset(data, video_root)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
        )
        return dataloader

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        self.optimizer.zero_grad()
        for step, (inputs, labels) in enumerate(
            tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.train_config.epochs}",
            ),
            1,
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            _, outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if step % 8 == 0 or step == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss /= len(self.train_loader)
        train_acc = 100.0 * correct_train / total_train
        return train_acc, train_loss

    def validate(self) -> tuple[float, float]:
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
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

    def save_model(self, val_loss: float, file_name: str | Path) -> None:
        save_model(
            self.model,
            asdict(self.train_config),
            val_loss,
            self.checkpoint_path / ("full_" + str(file_name)),
        )
        save_model(
            self.model.base,
            asdict(self.train_config),
            val_loss,
            self.checkpoint_path / ("base_" + str(file_name)),
        )

    def train(self) -> tuple[float, float, float, float]:
        self.criterion = nn.CrossEntropyLoss()

        # self.optimizer = optim.Adam(
        # self.model.parameters(),
        # lr=self.train_config.lr,
        # weight_decay=self.train_config.weight_decay,
        # )
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.train_config.lr,
            momentum=0.9,
            weight_decay=self.train_config.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
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
            scheduler.step(val_loss)
            logger.info(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%"
            )

            # Check early stopping condition
            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                logger.info("Best Model. Saving ...")
                self.save_model(val_loss, f"best_{self.model.base.name}.pt")
                best_train_acc, best_test_acc = train_acc, val_acc

            if early_stopping.early_stop and self.train_config.enable_earlystop:
                logger.warning(
                    f"Early stopping triggered. Stopping training with best_loss: {early_stopping.best_loss:.4f}."
                )
                break

        # Save final model
        self.save_model(val_loss, f"final_{self.model.base.name}.pt")
        return best_train_acc, best_test_acc, train_acc, val_acc


if __name__ == "__main__":
    config = load_config("Train and validate video classification model")

    logger.debug(f"Num Classes: {config.n_words}")

    model = Extractor(
        num_classes=config.n_words,
        base_model=config.extractor.model,
        n_freeze=config.extractor.training.freeze,
    )

    trainer = Trainer(
        config.data.processed.csvs.train,
        config.data.processed.csvs.val,
        config.data.processed.videos,
        model,
        config.extractor.training,
        config.extractor.checkpoints,
    )
    trainer.train()
