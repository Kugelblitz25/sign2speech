import torch
import torch.nn as nn


class SpectrogramGenerator(nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dims: list[int] = [1024, 2048, 2048]
    ) -> None:
        super(SpectrogramGenerator, self).__init__()
        mlp_layers = []
        current_dim = input_dim

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
        self.mlp.append(nn.Linear(current_dim, 128 * 5 * 5))

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, stride=2, padding=1
            ),  # (128, 10, 10)
            nn.LeakyReLU(),
            nn.InstanceNorm2d(128),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # (64, 20, 20)
            nn.LeakyReLU(),
            nn.InstanceNorm2d(64),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # (32, 40, 40)
            nn.LeakyReLU(),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(
                32, 16, kernel_size=(7, 3), stride=(5, 1), padding=(1, 1)
            ),  # (16, 200, 40)
            nn.LeakyReLU(),
            nn.InstanceNorm2d(16),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(
                16, 1, kernel_size=(32, 4), stride=(5, 2), padding=(1, 1)
            ),  # (1, 1025, 80)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.mlp(x)
        x = x.view(-1, 128, 5, 5)
        x = self.deconv_layers(x)
        return x
