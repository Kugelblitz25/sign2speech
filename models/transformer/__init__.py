from models.transformer.model import SpectrogramGenerator
from utils import load_model_weights
import torch

class FeatureTransformer:
    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpectrogramGenerator().to(self.device)
        self.model = load_model_weights(self.model, weights_path)

    def __call__(self, features):
        self.model.eval()
        with torch.no_grad():
            spec = self.model(features)
        return spec