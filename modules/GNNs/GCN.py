from ..layer.GCNconvCell import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from ..layer.GCNconvCell import MyGCN

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout=32, dropout=0.2):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # ¹¹½¨µÚÒ»²ãGCN
        self.gc2 = GraphConvolution(nhid, nout)  # ¹¹½¨µÚ¶þ²ãGCN
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GCN_gem(torch.nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN_gem, self).__init__()
        self.conv1 = GCNConv(nfeat, 32)
        self.conv2 = GCNConv(32, nhid)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        return x

class GCNconv(torch.nn.Module):
    def __init__(self, nfeat, nhid, n_out):
        super(GCNconv, self).__init__()
        self.conv1 = MyGCN(nfeat, nhid)
        self.conv2 = MyGCN(nhid, n_out)

    def forward(self, data):
        x, adj = data

        x, _ = self.conv1(x, adj)
        x = torch.relu(x)
        x, _ = self.conv2(x, adj)

        return x
