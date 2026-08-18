# -*- coding: utf-8 -*-
"""model.py — ONE canonical GraphSAGE_LinkPredictor.

All six definitions in the original file were identical in architecture.
This is the definitive version using PyG's SAGEConv.

Supports both 2-layer (hidden, out) and 3-layer (hidden1, hidden2, out)
configurations via the GNN_DIMS list.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import SAGEConv


class GraphSAGE_LinkPredictor(nn.Module):
    """2- or 3-layer GraphSAGE for link prediction (dot-product decoder)."""

    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 hidden_channels2: Optional[int] = None):
        """
        Parameters
        ----------
        in_channels : feature dimension
        hidden_channels : first hidden layer
        out_channels : final embedding dimension
        hidden_channels2 : optional second hidden layer (for 3-layer configs)
        """
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        if hidden_channels2 is not None:
            self.conv2 = SAGEConv(hidden_channels, hidden_channels2)
            self.conv3 = SAGEConv(hidden_channels2, out_channels)
            self._n_layers = 3
        else:
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            self._n_layers = 2

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        if self._n_layers == 3:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv3(x, edge_index)
        else:
            x = self.conv2(x, edge_index)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)
