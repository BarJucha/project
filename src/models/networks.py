"""Neural network architectures for decentralized graph navigation.

Implements ActorNetwork, CriticNetwork, and GATEncoder matching
Pisacane et al. (LoG 2024).
"""
import os
from typing import Optional
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATLayer(nn.Module):
    """GATv2 convolution followed by LayerNorm."""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=True)
        self.ln = nn.LayerNorm(out_channels * heads)

    def forward(self, x: T.Tensor, edge_index: T.Tensor) -> T.Tensor:
        return self.ln(self.conv(x, edge_index))


class GATEncoder(nn.Module):
    """3-layer Graph Attention Network for encoding local 1-hop ego-graphs."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 3,
        heads: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Layer 1: in_channels -> hidden_channels
        self.layers.append(GATLayer(in_channels, hidden_channels, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(heads * hidden_channels, hidden_channels, heads=heads))
            
        # Final layer: hidden_channels -> out_channels
        self.layers.append(GATLayer(heads * hidden_channels, out_channels, heads=1))

    def forward(self, x: T.Tensor, edge_index: T.Tensor) -> T.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


class ActorNetwork(nn.Module):
    """3-layer MLP Policy Network parameterizing pi(u -> u' | obs)."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        softmax_temp: float = 1.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.softmax_temp = max(softmax_temp, 1e-4)

    def forward(self, x: T.Tensor) -> T.Tensor:
        # x shape: (num_candidates, in_dim)
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))
        logits = self.fc3(h).squeeze(-1) / self.softmax_temp
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_logits(self, x: T.Tensor) -> T.Tensor:
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))
        return self.fc3(h).squeeze(-1) / self.softmax_temp


class CriticNetwork(nn.Module):
    """3-layer MLP Value Network estimating v(u, u_tgt)."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: T.Tensor) -> T.Tensor:
        h = F.relu(self.ln1(self.fc1(x)))
        h = F.relu(self.ln2(self.fc2(h)))
        out = self.fc3(h)
        return F.sigmoid(out).squeeze(-1)
