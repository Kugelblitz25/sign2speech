import torch
import torch.nn as nn

class SpectrogramGenerator(nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dims: list[int] = [1024, 512, 256, 1600], max_len: int = 320
    ) -> None:
<<<<<<< HEAD
        super().__init__()
=======
        super(SpectrogramGenerator, self).__init__()
        self.max_len = max_len

>>>>>>> d2f1c6add2976bfdb060ea10cec7bfb2cff4f56e
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


<<<<<<< HEAD
        self.abs = nn.Sequential(
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
                16, 1, kernel_size=(32, spec_len - 37), stride=(5, 1), padding=(1, 1)
            )
        )

        self.angle = nn.Sequential(
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
                16, 1, kernel_size=(32, spec_len - 37), stride=(5, 1), padding=(1, 1)
            )
=======
        base_height = 5  
        base_width = max_len // 16 
        self.base_shape = (32, base_height, base_width)  

        self.mlp = nn.Sequential(
            *mlp_layers,
            nn.Linear(current_dim, 32 * base_height * base_width)  
>>>>>>> d2f1c6add2976bfdb060ea10cec7bfb2cff4f56e
        )

       
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), 
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  
            nn.Conv2d(1, 1, kernel_size=(1, 1))  
        )

    def forward(self, x):
        x = self.mlp(x)
<<<<<<< HEAD
        x = x.view(-1, 128, 5, 5)
        abs = self.abs(x)
        theta = self.angle(x)
        spec = torch.concat([abs, theta], axis=1)
        return spec
=======
        x = x.view(x.shape[0], *self.base_shape) 
        x = self.deconv_layers(x)
        return x
>>>>>>> d2f1c6add2976bfdb060ea10cec7bfb2cff4f56e
