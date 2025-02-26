import time
from pathlib import Path

import pandas as pd
import torch
from speechbrain.inference.TTS import Tacotron2

from models.extractor.model import Extractor
from models.extractor.preprocessing.augmentation import main as augment_dataset
from models.extractor.preprocessing.verify import main as verify_videos
from models.extractor.train import Trainer
from models.generator.preprocessing.spec_gen import main as generate_specs
from utils.common import create_path, get_logger
from utils.config import load_config
from utils.model import load_model_weights

logger = get_logger("logs/inc_train.log")
config = load_config("Incremental Training")
checkpoint_path = create_path(config.generator.checkpoints)
specs_path = create_path(config.data.processed.specs)
classlist_path = create_path(config.data.processed.classlist)
output_video_dir = create_path(config.data.processed.videos)
device = "cuda" if torch.cuda.is_available() else "cpu"

tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir=checkpoint_path / "tts-tacotron2",
    run_opts={"device": device},
)


n_words_list = [10, 50, 100, 200, 300, 400, 500]
df = pd.DataFrame(columns=["n_words", "train_acc", "val_acc"])

for n_words in n_words_list:
    logger.info(f"Processing {n_words} words")
    t1 = time.perf_counter()
    generate_specs(
        config.data.raw.csvs.train, specs_path, classlist_path, n_words, tacotron2
    )
    verify_videos(
        config.data.raw.csvs,
        classlist_path,
        config.data.raw.videos,
        config.data.processed.csvs,
    )

    augment_dataset(
        config.data.processed.csvs,
        config.data.raw.videos,
        output_video_dir,
        config.extractor.num_augmentations,
    )
    model = Extractor(
        num_classes=n_words,
        base_model=config.extractor.model,
        n_freeze=config.extractor.training.freeze,
    )
    load_model_weights(
        model.base,
        Path(config.extractor.checkpoints) / f"base_best_{config.extractor.model}.pt",
        device,
    )

    trainer = Trainer(
        config.data.processed.csvs.train,
        config.data.processed.csvs.val,
        config.data.processed.videos,
        model,
        config.extractor.training,
        config.extractor.checkpoints,
    )
    train_acc, val_acc, *_ = trainer.train()
    t2 = time.perf_counter()
    logger.info(f"Time taken: {t2 - t1:.2f}s")
    logger.info(f"Train Acc: {train_acc}, Val Acc: {val_acc}")
    df = df.append(
        {"n_words": n_words, "train_acc": train_acc, "val_acc": val_acc},
        ignore_index=True,
    )
    df.to_csv("experiments/incremental_training.csv", index=False)
