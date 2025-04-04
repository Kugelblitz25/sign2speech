import torch
import torch.nn as nn

from models.extractor.model import Extractor
from models.transformer.model import SpectrogramGenerator


class CombinedModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model: str = "i3d",
        n_freeze: int = 0,
        hidden_dims: list[int] = [648, 1296, 2592, 5184],
        spec_len=64,
    ):
        super().__init__()

        self.extractor = Extractor(num_classes, model, n_freeze)
        self.transformer = SpectrogramGenerator(
            self.extractor.base.output_dim, hidden_dims, spec_len
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, logits = self.extractor(x)
        specs = self.transformer(features)
        return logits, specs
