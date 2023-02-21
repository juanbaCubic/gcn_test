import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch

class GraphSAGE(nn.Module):
    def __init__(self, data, dropout=0.2):
        super().__init__()

        self.data = data
        in_dim = self.data.num_features
        hidden_dim = 16
        out_dim = self.data.num_classes

        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)

    def forward(self):
        x = self.conv1(self.data.x, self.data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv2(x, self.data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv3(x, self.data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)