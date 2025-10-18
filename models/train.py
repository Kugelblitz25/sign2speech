from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dataset import S2S_Dataset
from models.model import S2S_Model
from models.transformer.train import complex_loss
from utils.common import create_path, create_subset, get_logger
from utils.config import CombinedTraining, load_config
from utils.model import EarlyStopping, load_model_weights, save_model

logger = get_logger("logs/combined_training.log")


class Trainer:
    def __init__(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path,
        video_root: str | Path,
        specs_csv: str | Path,
        model: S2S_Model,
        spec_len: int,
        train_cfg: CombinedTraining,
        checkpoint_path: str | Path,
        device: torch.device,
    ) -> None:
        self.device = device
        self.train_cfg = train_cfg
        self.checkpoint_path = create_path(checkpoint_path)

        logger.debug(f"Using device: {self.device}")

        self.train_loader = self.get_dataloader(
            train_data_path, specs_csv, video_root, spec_len
        )
        self.val_loader = self.get_dataloader(
            val_data_path, specs_csv, video_root, spec_len
        )

        self.model = model
        logger.info("Model initialized.")

    def get_dataloader(
        self,
        videos_csv: str | Path,
        specs_csv: str | Path,
        video_dir: str | Path,
        spec_len: int,
    ) -> DataLoader:
        dataset = S2S_Dataset(videos_csv, specs_csv, video_dir, spec_len)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
        )
        return dataloader

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        batch = 1

        mse = 0.0

        for video, spec in tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.train_cfg.epochs}"
        ):
            video, spec = (
                video.to(self.device),
                spec.to(self.device),
            )

            pred_spec = self.model(video)
            loss_value = complex_loss(spec, pred_spec)
            loss_value.backward()

            if batch % 8 == 0 or batch == num_batches:
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch += 1

            total_loss += loss_value.item()
            mse += F.mse_loss(spec, pred_spec).item()

        avg_loss = total_loss / num_batches
        avg_mse = mse / num_batches

        return avg_loss, avg_mse

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0

        mse = 0.0

        with torch.no_grad():
            for video, spec in tqdm(self.val_loader, desc="Validation"):
                video, spec = (
                    video.to(self.device),
                    spec.to(self.device),
                )

                pred_spec = self.model(video)
                loss_value = complex_loss(spec, pred_spec)

                total_loss += loss_value.item()
                mse += F.mse_loss(spec, pred_spec).item()

        avg_loss = total_loss / len(self.val_loader)
        avg_mse = mse / len(self.val_loader)

        return avg_loss, avg_mse

    def train(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg.epochs
        )

        early_stopping = EarlyStopping(patience=self.train_cfg.patience, verbose=True)

        logger.info("Starting training...")

        best_val_mse = 0.0

        for epoch in range(self.train_cfg.epochs):
            train_loss, train_mse = self.train_epoch(epoch)
            val_loss, val_mse = self.validate()

            logger.info(
                f"Epoch {epoch + 1}/{self.train_cfg.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}"
            )

            self.scheduler.step()

            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                logger.info("Best model so far. Saving...")
                best_val_mse = val_mse
                save_model(
                    self.model.extractor,
                    asdict(self.train_cfg),
                    val_loss,
                    self.checkpoint_path / "extractor_best.pt",
                )
                save_model(
                    self.model.transformer,
                    asdict(self.train_cfg),
                    val_loss,
                    self.checkpoint_path / "transformer_best.pt",
                )
            if early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best val MSE: {best_val_mse:.2f}")

        save_model(
            self.model.extractor,
            asdict(self.train_cfg),
            val_loss,
            self.checkpoint_path / "extractor_final.pt",
        )
        save_model(
            self.model.transformer,
            asdict(self.train_cfg),
            val_loss,
            self.checkpoint_path / "transformer_final.pt",
        )


if __name__ == "__main__":
    config = load_config(
        "Combined Feature Extraction and Spectrogram Generation",
        extractor_weights_path={
            "type": str,
            "default": None,
            "help": "Path to the extractor weights for fine-tuning",
        },
        transformer_weights_path={
            "type": str,
            "default": None,
            "help": "Path to the transformer weights for fine-tuning",
        },
        use_pretrained_base={
            "type": bool,
            "default": False,
            "help": "Whether the extractor weights given is for the base model only",
        },
    )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_data = create_subset(config.data.processed.csvs.train, config.n_words)
    val_data = create_subset(config.data.processed.csvs.val, config.n_words)

    model = S2S_Model(
        config.n_words,
        config.extractor.model,
        config.extractor.training.freeze,
        spec_len=config.generator.max_length,
    ).to(device)

    if config.extractor_weights_path:
        if config.use_pretrained_base:
            load_model_weights(
                model.extractor.base, config.extractor_weights_path, device
            )
        else:
            load_model_weights(model.extractor, config.extractor_weights_path, device)

    if config.transformer_weights_path:
        load_model_weights(model.transformer, config.transformer_weights_path, device)

    trainer = Trainer(
        train_data_path=train_data,
        val_data_path=val_data,
        video_root=config.data.processed.videos,
        specs_csv=config.data.processed.specs,
        model=model,
        spec_len=config.generator.max_length,
        train_cfg=config.combined.training,
        checkpoint_path=config.combined.checkpoints,
        device=device,
    )

    trainer.train()
