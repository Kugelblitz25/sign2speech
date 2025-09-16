import torch
import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50, r2plus1d_r50, x3d_s


class BaseExtractor(nn.Module):
    def __init__(self, model: str = "i3d", n_freeze: int = 0):
        super().__init__()
        self.name = model
        self.n_freeze = n_freeze
        match model:
            case "i3d":
                self.model = i3d_r50(pretrained=True)
                self.output_dim = 2048
            case "x3d":
                self.model = x3d_s(pretrained=True)
                self.output_dim = 192
            case "r2plus1d":
                self.model = r2plus1d_r50(pretrained=True)
                self.output_dim = 2048
            case _:
                raise ValueError(f"Invalid model name: {model}")

        self.model.blocks[-1] = nn.Sequential(
            nn.AvgPool3d(kernel_size=(4, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)),
        )

        for layer in self.model.blocks[: self.n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).permute(0, 2, 3, 4, 1)


class Extractor(nn.Module):
    def __init__(
        self, num_classes: int = 2000, base_model: str = "i3d", n_freeze: int = 0
    ) -> None:
        super().__init__()
        self.base = BaseExtractor(model=base_model, n_freeze=n_freeze)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5, inplace=True),
            nn.Linear(self.base.output_dim, num_classes, bias=True),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.base(x)
        output = self.classifier(features).permute(0, 4, 1, 2, 3)
        return self.flatten(features.permute(0, 4, 1, 2, 3)), self.flatten(output)
