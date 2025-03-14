import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from models.extractor.model import Extractor
from models.extractor.train import Trainer
from utils.common import get_logger
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/inc_train.log")
config = load_config("Incremental Training")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

n_words_list = [10, 50, 100, 200, 300, 400, 500]
df_path = "experiments/incremental_training/incremental_training.csv"

if os.path.exists(df_path):
    df = pd.read_csv(df_path)
else:
    df = pd.DataFrame(columns=["n_words", "train_acc", "val_acc"])

train_data = pd.read_csv(config.data.processed.csvs.train)
val_data = pd.read_csv(config.data.processed.csvs.val)

freq = dict(Counter(train_data.Gloss.to_list()))
words = sorted(freq, key=lambda x: -freq[x])

for n_words in n_words_list:
    logger.info(f"Processing {n_words} words")
    t1 = time.perf_counter()

    top_n_words = words[:n_words]
    n_train = train_data[train_data.Gloss.isin(top_n_words)]
    n_val = val_data[val_data.Gloss.isin(top_n_words)]

    n_train.to_csv("experiments/incremental_training/temp_train.csv", index=False)
    n_val.to_csv("experiments/incremental_training/temp_val.csv", index=False)

    model = Extractor(
        num_classes=n_words,
        base_model=config.extractor.model,
        n_freeze=config.extractor.training.freeze,
    )

    best_weights_path = (
        Path(config.extractor.checkpoints) / f"base_best_{config.extractor.model}.pt"
    )
    if best_weights_path.exists():
        load_model_weights(
            model.base,
            best_weights_path,
            torch.device(device),
        )

    trainer = Trainer(
        "experiments/incremental_training/temp_train.csv",
        "experiments/incremental_training/temp_val.csv",
        config.data.processed.videos,
        model,
        config.extractor.training,
        config.extractor.checkpoints,
    )

    train_acc, val_acc, *_ = trainer.train()
    t2 = time.perf_counter()
    logger.info(f"Time taken: {t2 - t1:.2f}s")
    logger.info(f"Train Acc: {train_acc}, Val Acc: {val_acc}")
    df.loc[len(df)] = [n_words, train_acc, val_acc]
    df.to_csv(df_path, index=False)
