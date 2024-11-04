import torch.nn as nn
import torch.nn.functional as F

class SpectrogramGenerator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512, 256, 1408]):
        super(SpectrogramGenerator, self).__init__()
        mlp_layers = []
        current_dim = input_dim

       # MLP layers
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim

        # Final MLP layer to match deconv input
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.append(nn.Linear(current_dim, 32 * 5 * 11))  # Adjusted to match ConvTranspose2d input

        # Deconvolution layers
        self.deconv_layers = nn.ModuleList([
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        ])

        # Batch normalization for deconvolution layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(16)
        ])

        # Final convolution layer
        self.final_conv = nn.Conv2d(1, 1, kernel_size=(1, 1))

    def forward(self, x):
        # MLP forward pass
        x = self.mlp(x)
        x = x.view(-1, 32, 5, 11)  # Reshape to match deconv input

        # Deconvolution forward pass
        for i, deconv in enumerate(self.deconv_layers[:-1]):
            x = self.batch_norms[i](deconv(x))
        
        x = self.deconv_layers[-1](x)  # Last deconv layer without batch norm
        x = self.final_conv(x)  # Final conv layer
        
        return x  # Expected output shape [batch_size, 1, 80, 88]
 
