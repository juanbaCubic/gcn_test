import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch

# GCN model with 2 layers
class GCN(nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()

        self.data = data

        self.conv1 = GCNConv(self.data.num_features, 16)
        self.conv2 = GCNConv(16, int(self.data.num_classes))

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, data, dropout=0.2):
        super(GraphSAGE, self).__init__()

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

class GAT(nn.Module):
    def __init__(self, data, heads=8):
        super(GAT, self).__init__()

        self.data = data
        in_dim = self.data.num_features
        hidden_dim = 16
        out_dim = self.data.num_classes

        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)

    def forward(self):

        h = F.dropout(self.data.x, p=0.6, training=self.training)
        h = self.gat1(h, self.data.edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, self.data.edge_index)
        return F.log_softmax(h, dim=1)
