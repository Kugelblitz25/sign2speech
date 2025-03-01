import numpy as np
import torch

from models.extractor import FeatureExtractor


class NMS:
    def __init__(
        self,
        extractor: FeatureExtractor,
        hop_length: int,
        win_size: int,
        overlap: int,
        threshold: float,
    ) -> None:
        self.extractor = extractor
        self.hop_length = hop_length
        self.win_size = win_size
        self.overlap = overlap
        self.threshold = threshold

    def predict(
        self, frames: list[np.ndarray]
    ) -> dict[int, tuple[torch.tensor, float]]:
        features = {}
        for i in range(0, len(frames), self.hop_length):
            ft, conf, _ = self.extractor(frames[i : i + self.win_size])
            features[i] = (ft, conf)
        return features

    def __call__(self, frames: list[np.ndarray]) -> dict[int, torch.tensor]:
        features = self.predict(frames)
        frame_idxs = [
            idx for idx, (_, prob) in features.items() if prob > self.threshold
        ]
        frame_idxs = sorted(frame_idxs, key=lambda x: features[x][1])
        good_preds = []
        while len(frame_idxs) > 0:
            frame_idx = frame_idxs.pop()
            good_preds.append(frame_idx)
            frame_idxs = [
                i
                for i in frame_idxs
                if abs(i - frame_idx) > self.win_size - self.overlap
            ]
        return {idx: features[idx][0] for idx in good_preds}
