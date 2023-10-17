from dgl.nn.pytorch import (TAGConv, GATConv, GINConv, GraphConv, GatedGraphConv,
                            MaxPooling, AvgPooling, SumPooling, GlobalAttentionPooling, SortPooling)
import torch.nn.functional as F
import torch


def create_activation_fn(act):
    if act == 'relu':
        return F.relu
    elif act == 'tanh':
        return F.tanh
    elif act == 'sigmoid':
        return torch.sigmoid
    elif act == 'gelu':
        return F.gelu
    else:
        return ValueError




def create_graph_pooling(pool):
    if pool == 'max':
        return MaxPooling
    elif pool == 'mean':
        return AvgPooling
    elif pool == 'sum':
        return SumPooling
