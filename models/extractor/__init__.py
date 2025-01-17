import numpy as np
import torch
import torch.nn.functional as F

from models.extractor.dataset import preprocess_video
from models.extractor.model import ModifiedI3D
from utils import load_model_weights


class FeatureExtractor:
    def __init__(self, weights_path: str, num_classes: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModifiedI3D(num_classes).to(self.device)
        self.model = load_model_weights(self.model, weights_path)

    def stack_frames(self, frames):
        frames_array = np.stack(frames, axis=0)
        frames_array = np.moveaxis(frames_array, -1, 0)
        return torch.tensor(frames_array)

    def __call__(self, frames: list[np.ndarray]):
        frames = self.stack_frames(frames)
        frames = preprocess_video(frames).to(self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            features, confs = self.model(frames)
            max_conf, idx = torch.max(F.softmax(confs, dim=1), 1)
        return features, max_conf, idx
