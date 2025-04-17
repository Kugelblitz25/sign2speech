import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.combined.dataset import CombinedDataset
from experiments.combined.model import CombinedModel
from utils.common import create_path, get_logger
from utils.config import load_config, ExtractorTraining, TransformerTraining
from utils.model import EarlyStopping, save_model, load_model_weights

logger = get_logger("logs/combined_training.log")


def spectral_convergence_loss(mel_true, mel_pred):
    return torch.norm(mel_true - mel_pred, p="fro") / torch.norm(mel_true, p="fro")


def complex_loss(true_complex, pred_complex, lambda_sc=0.5):
    # Extract real and imaginary parts
    true_real, true_imag = true_complex[:, 0:1], true_complex[:, 1:2]
    pred_real, pred_imag = pred_complex[:, 0:1], pred_complex[:, 1:2]

    # L1 loss for both components
    l1_real = F.l1_loss(pred_real, true_real)
    l1_imag = F.l1_loss(pred_imag, true_imag)

    # Spectral convergence for both components
    sc_real = spectral_convergence_loss(true_real, pred_real)
    sc_imag = spectral_convergence_loss(true_imag, pred_imag)

    # Reconstruction of magnitude
    true_mag = torch.sqrt(true_real**2 + true_imag**2)
    pred_mag = torch.sqrt(pred_real**2 + pred_imag**2)
    mag_loss = F.l1_loss(pred_mag, true_mag)

    # Combined loss
    return 2 * (l1_real + l1_imag + mag_loss) + lambda_sc * (sc_real + sc_imag)


def combined_loss(true_label, logits, true_spec, pred_spec, spec_weight=0.8):
    cls_loss = F.cross_entropy(logits, true_label)
    spec_loss = complex_loss(true_spec, pred_spec)
    return cls_loss + spec_weight * spec_loss


class CombinedOptimizer:
    def __init__(self, extractor_optim, transformer_optim):
        self.extractor_optim = extractor_optim
        self.transformer_optim = transformer_optim

    def zero_grad(self):
        self.extractor_optim.zero_grad()
        self.transformer_optim.zero_grad()

    def step(self):
        self.extractor_optim.step()
        self.transformer_optim.step()


class CombinedScheduler:
    def __init__(self, extractor_scheduler, transformer_scheduler):
        self.extractor_scheduler = extractor_scheduler
        self.transformer_scheduler = transformer_scheduler

    def step(self, val_loss):
        self.extractor_scheduler.step(val_loss)
        self.transformer_scheduler.step()


class CombinedTrainer:
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        video_root: str,
        specs_csv: str,
        model: CombinedModel,
        spec_len: int,
        extractor_config: ExtractorTraining,
        transformer_config: TransformerTraining,
        combined_checkpoint_path: str,
    ) -> None:
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.extractor_config = extractor_config
        self.transformer_config = transformer_config
        self.combined_checkpoint_path = create_path(combined_checkpoint_path)
        self.model = model.to(self.device)

        logger.debug(f"Using Device: {self.device}")

        # Load data
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        specs_df = pd.read_csv(specs_csv)

        # Create dataloaders
        self.train_loader = self.get_dataloader(
            train_data, video_root, specs_df, spec_len, True
        )
        self.val_loader = self.get_dataloader(
            val_data, video_root, specs_df, spec_len, False
        )

    def get_dataloader(
        self,
        data: pd.DataFrame,
        video_root: str,
        specs_df: pd.DataFrame,
        spec_len: int,
        is_train: bool,
    ) -> DataLoader:
        dataset = CombinedDataset(data, video_root, specs_df, spec_len)
        dataloader = DataLoader(
            dataset,
            batch_size=self.extractor_config.batch_size,
            shuffle=is_train,
            num_workers=self.extractor_config.num_workers,
        )
        return dataloader

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for videos, labels, spectrograms in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs}",
        ):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            spectrograms = spectrograms.to(self.device)

            logits, pred_specs = self.model(videos)
            loss = self.criterion(labels, logits, spectrograms, pred_specs)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(F.softmax(logits, dim=1), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        return (
            total_loss / len(self.train_loader),
            train_acc,
        )

    def validate(self) -> tuple[float, float, float, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels, spectrograms in tqdm(
                self.val_loader, desc="Validation"
            ):
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                spectrograms = spectrograms.to(self.device)

                logits, pred_specs = self.model(videos)
                loss = self.criterion(labels, logits, spectrograms, pred_specs)

                # Track metrics
                total_loss += loss.item()

                _, predicted = torch.max(F.softmax(logits, dim=1), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / total
        return (
            total_loss / len(self.val_loader),
            val_acc,
        )

    def train(self, epochs: int = 500) -> None:
        self.criterion = combined_loss

        extractor_optim = optim.SGD(
            self.model.extractor.parameters(),
            lr=self.extractor_config.lr,
            momentum=0.9,
            weight_decay=self.extractor_config.weight_decay,
        )
        transformer_optim = optim.Adam(
            self.model.transformer.parameters(),
            lr=self.transformer_config.lr,
            weight_decay=self.transformer_config.weight_decay,
        )

        self.optimizer = CombinedOptimizer(extractor_optim, transformer_optim)

        extractor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.extractor_optim,
            mode="min",
            factor=self.extractor_config.scheduler_factor,
            patience=self.extractor_config.scheduler_factor,
        )
        transformer_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer.transformer_optim, 50
        )

        scheduler = CombinedScheduler(extractor_scheduler, transformer_scheduler)

        early_stopping = EarlyStopping(
            patience=self.transformer_config.patience, verbose=True
        )

        logger.info(f"Starting combined training.")

        self.epochs = epochs

        best_val_acc = 0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            scheduler.step(val_loss)
            logger.info(
                f"Epoch {epoch + 1}: "
                f"Train [Loss={train_loss:.4f}, Acc={train_acc:.2f}%], "
                f"Val [Loss={val_loss:.4f}, Acc={val_acc:.2f}%]"
            )

            # Check early stopping
            early_stopping(val_loss)
            if early_stopping.best_loss == val_loss:
                logger.info("Best model so far. Saving...")
                best_val_acc = val_acc

                # Save both models
                save_model(
                    self.model.extractor,
                    self.extractor_config,
                    val_loss,
                    self.combined_checkpoint_path / "extractor_best.pt",
                )

                save_model(
                    self.model.transformer,
                    self.transformer_config,
                    val_loss,
                    self.combined_checkpoint_path / "generator_best.pt",
                )

            if early_stopping.early_stop:
                logger.warning(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Save final models
        logger.info(
            f"Training completed. Best validation accuracy: {best_val_acc:.2f}%"
        )
        save_model(
            self.model.extractor,
            self.extractor_config,
            val_loss,
            self.combined_checkpoint_path / "extractor_best.pt",
        )

        save_model(
            self.model.transformer,
            self.transformer_config,
            val_loss,
            self.combined_checkpoint_path / "generator_best.pt",
        )


if __name__ == "__main__":
    config = load_config("Combined Feature Extraction and Spectrogram Generation")

    model = CombinedModel(
        num_classes=config.n_words,
        model=config.extractor.model,
        n_freeze=config.extractor.training.freeze,
        spec_len=config.generator.max_length,
    )

    # load_model_weights(model.extractor.base, "models/extractor/checkpoints/base_best_i3d_1500.pt", "cuda:1")

    trainer = CombinedTrainer(
        train_data_path=config.data.processed.csvs.train,
        val_data_path=config.data.processed.csvs.val,
        video_root=config.data.processed.videos,
        specs_csv=config.data.processed.specs,
        model=model,
        spec_len=config.generator.max_length,
        extractor_config=config.extractor.training,
        transformer_config=config.transformer.training,
        combined_checkpoint_path="experiments/combined/checkpoints",
    )

    trainer.train()
