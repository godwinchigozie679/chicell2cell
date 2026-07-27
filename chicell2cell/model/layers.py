"""Neural network layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .config import FLAGS


class ExpRec(nn.Module):
    def __init__(self, n_features, latent_dim, dropout=0.7):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.reconstruction_layer = nn.Linear(latent_dim, n_features)
        nn.init.xavier_uniform_(self.reconstruction_layer.weight, gain=0.1)
        nn.init.zeros_(self.reconstruction_layer.bias)

    def forward(self, z):
        z = self.dropout_layer(z)
        return self.reconstruction_layer(z)


class GATVAELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_heads=1, dropout=0.7):
        super().__init__()
        self.dropout = dropout
        self.gat1    = GATConv(input_dim, hidden_dim, heads=num_heads,
                               dropout=dropout, concat=True)
        self.bn1     = nn.BatchNorm1d(hidden_dim * num_heads)
        self.gat2    = GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                               dropout=dropout, concat=False)
        self.bn2     = nn.BatchNorm1d(hidden_dim)
        self.z_mean     = nn.Linear(hidden_dim, latent_dim)
        self.z_log_std  = nn.Linear(hidden_dim, latent_dim)

        nn.init.xavier_uniform_(self.z_mean.weight,    gain=0.1)
        nn.init.zeros_(self.z_mean.bias)
        nn.init.xavier_uniform_(self.z_log_std.weight, gain=0.1)
        nn.init.constant_(self.z_log_std.bias, -2.0)

        # Cached attention weights — updated on every forward pass
        self.attention_weights        = None
        self.edge_index_with_attention = None

    def encode(self, x, edge_index):
        if self.training and FLAGS.noise_std > 0:
            x = x + torch.randn_like(x) * FLAGS.noise_std

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x, (edge_index_att, attn_w) = self.gat2(
            x, edge_index, return_attention_weights=True
        )
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        self.attention_weights         = attn_w
        self.edge_index_with_attention = edge_index_att

        return self.z_mean(x), self.z_log_std(x)

    def reparameterize(self, z_mean, z_log_std):
        if self.training:
            std = torch.exp(z_log_std)
            return z_mean + torch.randn_like(std) * std
        return z_mean

    def forward(self, x, edge_index):
        z_mean, z_log_std = self.encode(x, edge_index)
        z = self.reparameterize(z_mean, z_log_std)
        if self.training:
            z = F.dropout(z, p=0.3, training=True)
        return z, z_mean, z_log_std


class Decoder(nn.Module):
    """
    Bilinear decoder: A_hat = sigmoid(Z W Z^T)
    Reference: Nickel et al. ICML 2011
    """
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

    def forward(self, z):
        z = F.dropout(z, p=self.dropout, training=self.training)
        return torch.sigmoid(z @ self.W @ z.t())
