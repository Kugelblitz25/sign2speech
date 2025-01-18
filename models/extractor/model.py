import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50


class ModifiedI3D(nn.Module):
    def __init__(self, num_classes: int = 2000):
        super(ModifiedI3D, self).__init__()
        self.i3d = i3d_r50(pretrained=True)
        self.i3d.blocks = self.i3d.blocks[:-1]

        self.features = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        conv = self.i3d(x)
        features = self.features(conv)
        output = self.classifier(features)
        return features, output
