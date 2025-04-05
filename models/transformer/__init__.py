import torch

from models.transformer.model import SpectrogramGenerator
from utils.model import load_model_weights


class FeatureTransformer:
    def __init__(self, weights_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpectrogramGenerator().to(self.device)
        load_model_weights(self.model, weights_path, self.device)

    def __call__(self, features: torch.tensor) -> torch.tensor:
        self.model.eval()
        with torch.no_grad():
            spec = self.model(features)
        return spec
