import time
from pathlib import Path

import torch

from models.extractor.model import Extractor
from models.extractor.train import Trainer
from utils.common import get_logger
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/wlasl_train.log")
config = load_config("WLASL Training")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

t1 = time.perf_counter()

model = Extractor(
    num_classes=config.n_words,
    base_model=config.extractor.model,
    n_freeze=config.extractor.training.freeze,
)

load_model_weights(
    model.base, Path(config.extractor.checkpoints) / "base_best_i3d.pt", device
)

trainer = Trainer(
    config.data.processed.csvs.train,
    config.data.processed.csvs.val,
    config.data.processed.videos,
    model,
    config.extractor.training,
    "experiments/wlasl",
)

train_acc, val_acc, *_ = trainer.train()
t2 = time.perf_counter()
logger.info(f"Time taken: {t2 - t1:.2f}s")
logger.info(f"Train Acc: {train_acc}, Val Acc: {val_acc}")
