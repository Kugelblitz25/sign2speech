import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from models.extractor.model import Extractor
from models.extractor.test import Tester
from utils.common import get_logger
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/standalone_test.log")
config = load_config("Standalone Training")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

n_words = 750

test_data = pd.read_csv(config.data.processed.csvs.test)
train_data = pd.read_csv(config.data.processed.csvs.train)

freq = dict(Counter(train_data.Gloss.to_list()))
words = sorted(freq, key=lambda x: -freq[x])

logger.info(f"Processing {n_words} words")
t1 = time.perf_counter()

top_n_words = words[:n_words]
n_test = test_data[test_data.Gloss.isin(top_n_words)]

n_test.to_csv("experiments/incremental_training/temp_test.csv", index=False)

model = Extractor(
    num_classes=n_words,
    base_model=config.extractor.model,
    n_freeze=config.extractor.training.freeze,
)

load_model_weights(
    model,
    Path(config.extractor.checkpoints) / f"full_best_{config.extractor.model}.pt",
    torch.device("cuda:1"),
)

tester = Tester(
    model=model,
    test_data_path="experiments/incremental_training/temp_test.csv",
    video_root=config.data.processed.videos,
    batch_size=config.extractor.training.batch_size,
    num_workers=config.extractor.training.num_workers,
)

results = tester.evaluate()
tester.print_results(results)
t2 = time.perf_counter()
logger.info(f"Time taken: {t2 - t1:.2f}s")
