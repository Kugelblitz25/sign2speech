import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramGenerator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512]):
        """
        Hybrid MLP-Deconvolutional network for generating mel spectrograms from video features.
        
        Args:
            input_dim (int): Input feature dimension (default: 2048)
            hidden_dims (list): Dimensions of hidden MLP layers
        """
        super(SpectrogramGenerator, self).__init__()
        
        # MLP layers
        mlp_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.append(nn.Linear(current_dim, 2816))
        
        # Deconvolutional layers
        self.deconv_layers = nn.ModuleList([
            # Initial shape after reshape: [batch, 32, 4, 11]
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 8, 22]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: [batch, 32, 16, 44]
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: [batch, 16, 32, 88]
            nn.ConvTranspose2d(16, 1, kernel_size=(4,3), stride=(2,1), padding=(1,1))  # Output: [batch, 1, 64, 88]
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(16)
        ])
        
        # Final 1x1 convolution to adjust to exact target size
        self.final_conv = nn.Conv2d(1, 1, kernel_size=(1, 1))
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2048]
            
        Returns:
            torch.Tensor: Generated mel spectrogram of shape [batch_size, 1, 88, 128]
        """
        # MLP layers
        x = self.mlp(x)
        x = x.view(-1, 32, 8, 11)
        
        # Deconvolutional layers
        for i, deconv in enumerate(self.deconv_layers[:-1]):
            x = deconv(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout2d(x, p=0.3, training=self.training)
        
        # Final deconv layer without batch norm
        x = self.deconv_layers[-1](x)
        x = self.final_conv(x)
        
        return torch.sigmoid(x)  # Sigmoid to ensure values are between 0 and 1
