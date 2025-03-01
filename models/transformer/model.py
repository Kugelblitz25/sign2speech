import torch
import torch.nn as nn


class SpectrogramGenerator(nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dims: list[int] = [1024, 512, 256, 1408]
    ) -> None:
        super(SpectrogramGenerator, self).__init__()
        mlp_layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.append(nn.Linear(current_dim, 32 * 5 * 11))

        self.deconv_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose2d(
                    16, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)
                ),
            ]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm2d(64), nn.BatchNorm2d(32), nn.BatchNorm2d(16)]
        )

        self.final_conv = nn.Conv2d(1, 1, kernel_size=(1, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.mlp(x)
        x = x.view(-1, 32, 5, 11)

        for i, deconv in enumerate(self.deconv_layers[:-1]):
            x = self.batch_norms[i](deconv(x))

        x = self.deconv_layers[-1](x)
        x = self.final_conv(x)

        return x
