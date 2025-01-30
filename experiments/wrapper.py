from collections import namedtuple

import pandas as pd

from models.extractor.train import Trainer
from utils import Config

cfg = Config("Experimenting with different freeze layers and models")

df = pd.DataFrame(
    columns=[
        "model",
        "freeze_blocks",
        "best_train_acc",
        "best_test_acc",
        "final_train_acc",
        "final_test_acc",
    ]
)

num_tries = 5
for model in ["i3d", "x3d"]:
    for freeze in range(6):
        best_train_acc, best_test_acc, final_train_acc, final_test_acc = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        train_cfg = cfg.extractor.training._asdict()
        train_cfg["freeze"] = freeze
        trainCFG = namedtuple("trainCFG", train_cfg.keys())
        train_cfg = trainCFG(**train_cfg)
        print(train_cfg)
        for _ in range(num_tries):
            trainer = Trainer(
                cfg.data.processed.csvs.train,
                cfg.data.processed.csvs.val,
                cfg.data.processed.videos,
                cfg.n_words,
                model,
                train_cfg,
                cfg.extractor.checkpoints,
            )
            a, b, c, d = trainer.train()
            best_train_acc += a
            best_test_acc += b
            final_train_acc += c
            final_test_acc += d
        df.loc[len(df)] = [
            model,
            freeze,
            best_train_acc / num_tries,
            best_test_acc / num_tries,
            final_train_acc / num_tries,
            final_test_acc / num_tries,
        ]

df.to_csv("experiments/freeze.csv", index=False)
