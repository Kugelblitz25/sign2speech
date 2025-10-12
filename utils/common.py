import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pandas as pd

type path = str | Path
type flist = list[np.ndarray] | np.ndarray
type source = path | flist


class Video:
    def __init__(self, source: source) -> None:
        if isinstance(source, (str, Path)):
            self.type = path
            self._init_from_path(source)
        elif isinstance(source, (list, np.ndarray)):
            self.type = flist
            self._init_from_flist(source)
        else:
            raise ValueError("Unsupported source type")

    def _init_from_path(self, source: path):
        self.video = cv2.VideoCapture(source)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _init_from_flist(self, source: flist):
        if isinstance(source, list):
            source = np.array(source)
        assert len(source.shape) == 4, "Flist must be a 4D array"
        assert source.shape[3] == 3, "Last dimension must be 3 (RGB)"
        self.video = source
        self.fps = 25
        self.n_frames = source.shape[0]
        self.width = source.shape[2]
        self.height = source.shape[1]

    def release(self):
        if self.type == path:
            self.video.release()

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        if self.type == flist:
            for frame in self.video:
                yield frame
            return
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    break
                yield frame
        finally:
            self.release()

    def __del__(self):
        self.release()


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
