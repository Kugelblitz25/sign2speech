import numpy as np
import torch
import torch.nn.functional as F

from models.extractor.dataset import transform
from models.extractor.model import Extractor
from utils.model import load_model_weights


class FeatureExtractor:
    def __init__(self, weights_path: str, num_classes: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Extractor(num_classes).to(self.device)
        load_model_weights(self.model, weights_path, self.device)

    def stack_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        frames_array = np.stack(frames, axis=0)
        frames_array = np.moveaxis(frames_array, -1, 0)
        return torch.tensor(frames_array)

    def __call__(self, frames: list[np.ndarray]) -> tuple[torch.Tensor, float, int]:
        frames = self.stack_frames(frames)
        frames = transform(frames).to(self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            features, confs = self.model(frames)
            max_conf, idx = torch.max(F.softmax(confs, dim=1), 1)
        return features, max_conf.cpu().numpy()[0], idx.cpu().numpy()[0]
