from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer.dataset import SpectrogramDataset
from models.transformer.model import SpectrogramGenerator
from utils.common import create_path, get_logger
from utils.config import TransformerTraining as TrainConfig
from utils.config import load_config
from utils.model import EarlyStopping, save_model

logger = get_logger("logs/transformer_training.log")

def spectral_convergence_loss(mel_true, mel_pred):
    return torch.norm(mel_true - mel_pred, p="fro") / torch.norm(mel_true, p="fro")


def combined_loss(mel_true, mel_pred, lambda_sc=0.1):
    l1 = nn.functional.l1_loss(mel_pred, mel_true)
    sc = spectral_convergence_loss(mel_true, mel_pred)
    return l1 + lambda_sc * sc


class Trainer:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        specs_csv: str,
        train_config: TrainConfig,
        checkpoint_path: str,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_config = train_config
        self.checkpoint_path = create_path(checkpoint_path)
        logger.debug(f"Using Device: {self.device}")

        self.model = SpectrogramGenerator().to(self.device)
        self.train_loader = self.get_dataloader(train_data_path, specs_csv)
        self.val_loader = self.get_dataloader(val_data_path, specs_csv)

    def get_dataloader(self, features_csv: str, spec_csv: str) -> DataLoader:
        dataset = SpectrogramDataset(features_csv, spec_csv)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
        )
        return dataloader

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0

        for features, spectrograms in tqdm(
            self.train_loader, f"Epoch {epoch}/{self.train_config.epochs}"
        ):
            features = features.to(self.device)
            spectrograms = spectrograms.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(spectrograms, outputs)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for features, spectrograms in tqdm(self.val_loader, "Validation"):
                features = features.to(self.device)
                spectrograms = spectrograms.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(spectrograms, outputs)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> None:
        self.criterion = combined_loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_config.lr,
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

        logger.critical("Started transformer training.")
        for epoch in range(self.train_config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            scheduler.step(val_loss)
            logger.info(
                f"Epoch: {epoch + 1} Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                logger.info("Best Model. Saving ...")
                save_model(
                    self.model,
                    asdict(self.train_config),
                    val_loss,
                    self.checkpoint_path / "checkpoint_best.pt",
                )

            if early_stopping.early_stop and self.train_config.enable_earlystop:
                logger.warning("Early stopping triggered.")
                break

        logger.warning(
            f"Stopped training with best_loss: {early_stopping.best_loss:.4f}."
        )
        save_model(
            self.model,
            asdict(self.train_config),
            val_loss,
            self.checkpoint_path / "checkpoint_final.pt",
        )


if __name__ == "__main__":
    config = load_config("Transforming video features into spectrogram features")

    trainer = Trainer(
        config.data.processed.vid_features.train,
        config.data.processed.vid_features.val,
        config.data.processed.specs,
        config.transformer.training,
        config.transformer.checkpoints,
    )
    trainer.train()
