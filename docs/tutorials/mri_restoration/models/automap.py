import torch
import torch.nn as nn
import torch.nn.functional as F


class Automap(nn.Module):
    def __init__(self, m, K, dim_bottleneck=64, conv_channels=64):
        """PyTorch implementation of AUTOMAP
        Zhu, B., Liu, J. Z., Cauley, S. F., Rosen, B. R., & Rosen, M. S. (2018). 
        Image reconstruction by domain-transform manifold learning. Nature, 555(7697), 487-492
        """
        super(Automap, self).__init__()
        self.fc1 = nn.Linear(2*m, dim_bottleneck)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(dim_bottleneck, K**2)
        self.layer_norm = nn.LayerNorm(K**2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=5, padding=2)
        self.tconv = nn.ConvTranspose2d(in_channels=conv_channels, out_channels=1, kernel_size=7, padding=3)
        
    def forward(self, kspace, mask):
        kspace = torch.view_as_real(kspace)
        x = torch.flatten(kspace, start_dim=1)
        x = F.tanh(self.fc1(x))
        x = self.layer_norm(self.fc2(x))
        x = x.view(-1, 1, self.K, self.K)
        x = F.tanh(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.tconv(x)
        return x.unsqueeze(-1)


'''
En version séquentielle :

class Automap(nn.Module):
    def __init__(self, m, K, dim_bottleneck=64, conv_channels=64):
        super(Automap, self).__init__()

        # Séquentiel dense + tanh
        self.dense = nn.Sequential(
            nn.Linear(2 * m, dim_bottleneck),
            nn.Tanh(),
            nn.Linear(dim_bottleneck, K * K),
            nn.LayerNorm(K * K)
        )

        # Séquentiel convolutionnel
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_channels, 1, kernel_size=7, padding=3)
        )

        self.K = K

    def forward(self, kspace, mask=None):
        # kspace: (batch, H, W), complexe
        x = torch.view_as_real(kspace)  # (batch, H, W, 2)
        x = x.reshape(x.shape[0], -1)   # (batch, 2*H*W)
        x = self.dense(x)               # (batch, K*K)
        x = x.view(-1, 1, self.K, self.K)  # (batch, 1, K, K)
        x = self.conv(x)                # (batch, 1, K, K)
        return x
'''
