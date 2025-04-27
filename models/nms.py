from collections import deque

import numpy as np
import torch

from models.extractor import FeatureExtractor
from utils.config import NMSConfig


class NMS:
    def __init__(self, extractor: FeatureExtractor, config: NMSConfig) -> None:
        self.extractor = extractor
        self.win_size = config.win_size
        self.overlap = config.overlap
        self.threshold = config.threshold
        self.hop_length = config.hop_length

        self.window = deque(maxlen=self.win_size)
        self.window_index = None
        self.best_window_idx = None
        self.best_feature = None
        self.best_confidence = 0.0

    def __call__(self, frame: np.ndarray) -> tuple[int, torch.Tensor | None]:
        self.window.append(frame)
        if len(self.window) < self.win_size:
            return -1, None

        if self.window_index is None and len(self.window) == self.win_size:
            feature, confidence = self.extractor(list(self.window))
            self.window_index = 0
            if confidence >= self.threshold:
                self.best_window_idx = 0
                self.best_feature = feature
                self.best_confidence = confidence
            return -1, None


        self.window_index += 1
        if not self.window_index % self.hop_length == 0:
            return -1, None

        feature, confidence = self.extractor(list(self.window))

        if self.best_window_idx is None and confidence > self.threshold:
            self.best_window_idx = self.window_index
            self.best_feature = feature
            self.best_confidence = confidence
            return -1, None

        if self.best_window_idx is None:
            return -1, None
            
        if (self.window_index - self.best_window_idx) > (self.win_size - self.overlap):
            return_index = self.best_window_idx
            return_feature = self.best_feature
            if confidence > self.threshold:
                self.best_window_idx = self.window_index
                self.best_feature = feature
                self.best_confidence = confidence
            else:
                self.best_window_idx = None
                self.best_feature = None
                self.best_confidence = 0.0
            return return_index, return_feature

        if confidence > self.best_confidence:
            self.best_window_idx = self.window_index
            self.best_feature = feature
            self.best_confidence = confidence

        return -1, None