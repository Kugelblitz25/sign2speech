import torch.nn as nn
from pytorchvideo.models.hub import i3d_r50, r2plus1d_r50, x3d_s


class ModifiedI3D(nn.Module):
    def __init__(self, num_classes: int = 2000):
        super().__init__()
        self.backbone = i3d_r50(pretrained=True)
        self.backbone.blocks = self.backbone.blocks[:-1]
        self.name = "i3d"

        self.features = nn.AvgPool3d(
            kernel_size=(4, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)
        )
        self.dropout = nn.Dropout(0.5, inplace=False)
        # nn.Linear(2048, 1024),
        # nn.BatchNorm1d(1024),
        # nn.ReLU(),
        # nn.Linear(1024, 512),
        # nn.BatchNorm1d(512),
        # nn.ReLU(),
        # nn.Dropout(0.6),
        # nn.Linear(512, 256),
        # nn.BatchNorm1d(256),
        # nn.ReLU(),
        # nn.Dropout(0.6),

        self.classifier = nn.Sequential(
            nn.Linear(2048, num_classes, bias=True),
        )
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        conv = self.backbone(x)
        features = self.features(conv)

        features = self.dropout(features).permute(0, 2, 3, 4, 1)
        output = self.classifier(features).permute(0, 4, 1, 2, 3)
        return features, self.flatten(output)


class ModifiedX3D(nn.Module):
    def __init__(self, num_classes: int = 2000):
        super().__init__()
        self.backbone = x3d_s(pretrained=True)
        self.backbone.blocks = self.backbone.blocks[:-1]
        self.name = "x3d"

        self.features = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(192, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.LayerNorm(512),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(512, 256),
            # nn.LayerNorm(256),
            # nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        conv = self.backbone(x)
        features = self.features(conv)
        output = self.classifier(features)
        return features, output


class ModifiedR2P1D(nn.Module):
    def __init__(self, num_classes: int = 2000):
        super().__init__()
        self.backbone = r2plus1d_r50(pretrained=True)
        self.backbone.blocks = self.backbone.blocks[:-1]
        self.name = "r2plus1d"

        self.features = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        conv = self.backbone(x)
        features = self.features(conv)
        output = self.classifier(features)
        return features, output
