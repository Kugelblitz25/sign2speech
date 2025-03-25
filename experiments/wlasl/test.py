from pathlib import Path
import time

import torch

from models.extractor.model import Extractor
from models.extractor.test import Tester
from utils.common import get_logger
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/wlasl_test.log")
config = load_config("WLASL Testing")
device = "cuda:1" if torch.cuda.is_available() else "cpu"

t1 = time.perf_counter()

model = Extractor(
    num_classes=config.n_words,
    base_model=config.extractor.model,
    n_freeze=config.extractor.training.freeze,
)

load_model_weights(model, "experiments/wlasl/full_best_i3d.pt", torch.device("cuda:1"))

tester = Tester(
    model=model,
    test_data_path=config.data.processed.csvs.test,
    video_root=config.data.processed.videos,
    batch_size=config.extractor.training.batch_size,
    num_workers=config.extractor.training.num_workers,
)

results = tester.evaluate()
tester.print_results(results, Path("experiments/wlasl/"))
t2 = time.perf_counter()
logger.info(f"Time taken: {t2 - t1:.2f}s")
