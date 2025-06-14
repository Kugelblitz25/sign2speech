import torch
import torch.nn as nn


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        dropout: float,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DeconvNetwork(nn.Module):
    def __init__(self, init_kernels: int, spec_len: int):
        super().__init__()

        self.network = nn.Sequential(
            DeconvBlock(init_kernels, 64, 3, (3, 2), (2, 2), 0.0),
            DeconvBlock(64, 32, 5, (3, 2), (3, 2), 0.0),
            DeconvBlock(32, 16, 7, (4, 2), (3, 2), 0.0),
            DeconvBlock(16, 8, (7, spec_len - 54), (4, 1), (3, 2), 0.0),
            nn.ConvTranspose2d(
                in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x):
        return self.network(x)


class SpectrogramGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: list[int] = [648, 1296, 2592, 5184],
        spec_len=64,
    ) -> None:
        super().__init__()
        mlp_layers = []
        current_dim = input_dim
        self.init_kernels = 128

        for hidden_dim in hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(0.1),
                ]
            )
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.append(nn.Linear(current_dim, self.init_kernels * 9 * 9))

        self.abs = DeconvNetwork(self.init_kernels, spec_len)
        self.angle = DeconvNetwork(self.init_kernels, spec_len)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.mlp(x)
        x = x.view(-1, self.init_kernels, 9, 9)
        abs = self.abs(x)
        theta = self.angle(x)
        spec = torch.concat([abs, theta], axis=1)
        return spec
