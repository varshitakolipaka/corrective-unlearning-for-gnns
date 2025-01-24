import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, return_all_emb=False, get_pre_final=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index)
        if return_all_emb:
            return x1, x2, x3
        if get_pre_final:
            return x2
        return x3

    def get_last_layer_emb(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        return x3

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.lin = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        x4 = self.lin(x3)
        if return_all_emb:
            return x1, x2, x3
        return x4

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
    
    def get_last_layer_emb(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        return x3

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(in_dim, hidden_dim))
        self.conv2= GINConv(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x2 = self.conv2(x, edge_index)
        if return_all_emb:
            return x1, x2
        return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

