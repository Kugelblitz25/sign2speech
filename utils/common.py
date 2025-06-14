import logging
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd


def get_logger(loc: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(create_path(loc))
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s", datefmt="%d/%m/%Y %H:%M"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.suffix:
        path.mkdir(exist_ok=True, parents=True)
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
    return path


def create_subset(source_file: str | Path, n_words: int) -> Path:
    data = pd.read_csv(source_file)
    freq = dict(Counter(data.Gloss.to_list()))
    words = sorted(freq, key=lambda x: -freq[x])
    top_n_words = words[:n_words]

    subset_data = data[data.Gloss.isin(top_n_words)]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        subset_data.to_csv(temp_file.name, index=False)
    return Path(temp_file.name)
