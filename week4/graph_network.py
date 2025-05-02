import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class LJGnn(MessagePassing):
    def __init__(self, hidden_dim=64):
        super().__init__(aggr='add')  
        self.scalar_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # edge_attr: [dx, dy, r]
        return self.propagate(edge_index, edge_attr=edge_attr, size=(x.size(0), x.size(0)))

    def message(self, edge_attr):
       
        rel_vec = edge_attr[:, :2]  # (dx, dy)
        dist = edge_attr[:, 2:3]    # r

        # Avoid divide-by-zero
        eps = 1e-8
        direction = rel_vec / (dist + eps) 
        scalar = self.scalar_net(dist)      
        return direction * scalar           

    def update(self, aggr_out):
        return aggr_out 
