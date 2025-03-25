import torch
import torch.nn as nn

class SpectrogramGenerator(nn.Module):
    def __init__(
        self, input_dim: int = 2048, hidden_dims: list[int] = [1024, 512, 256, 1600], max_len: int = 320
    ) -> None:
        super(SpectrogramGenerator, self).__init__()
        self.max_len = max_len

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


        base_height = 5  
        base_width = max_len // 16 
        self.base_shape = (32, base_height, base_width)  

        self.mlp = nn.Sequential(
            *mlp_layers,
            nn.Linear(current_dim, 32 * base_height * base_width)  
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
        x = x.view(x.shape[0], *self.base_shape) 
        x = self.deconv_layers(x)
        return x