"""
"""

from ..utils import LayerFactory, ActivationFactory
import torch
from torch_geometric.nn import GCNConv, GATConv, TAGConv, ChebConv, SAGEConv, GatedGraphConv, ARMAConv, GraphConv, GraphUNet, AGNNConv, APPNP, GINConv
import torch.nn as nn
import torch.nn.functional as F
import warnings
from deepdrugdomain.utils import MissingRequiredParameterError
from torch_geometric.typing import Adj
from torch_geometric.data import Data


@LayerFactory.register('pyg_gcn')
class GCN(nn.Module):

    def __init__(self, in_feat, out_feat, normalization=True, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "improved": False,
            "cached": False,
            "add_self_loops": True,
            "bias": True,
            'activation': "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()

        self.layer = GCNConv(
            in_channels=in_feat, out_channels=out_feat, normalize=normalization, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_attr if 'edge_attr' in graph else None

        features = self.activation(self.layer(
            features, edge_index, edge_weight))
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=edge_weight)

        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_gat')
class GAT(nn.Module):
    """
    Wrapper class for DGL's Graph Attention Network (GAT) layer.
    """

    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "heads": 1,
            "concat": True,
            "negative_slope": 0.2,
            "dropout": 0.0,
            "add_self_loops": True,
            "edge_dim": None,
            "fill_value": 'mean',
            "bias": True,
            "activation": "relu",
        }
        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()

        self.layer = GATConv(in_channels=in_feat,
                             out_channels=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_attr if 'edge_attr' in graph else None

        features = self.activation(self.layer(
            features, edge_index, edge_weight))
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=edge_weight)
        del features, edge_index, edge_weight, graph
        return new_graph


@LayerFactory.register('pyg_tag')
class TAG(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=True, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "K": 3,
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()

        self.layer = TAGConv(
            in_channels=in_feat, out_channels=out_feat, normalize=normalization, **kwargs)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_attr if 'edge_attr' in graph else None

        features = self.activation(self.layer(
            features, edge_index, edge_weight))
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=edge_weight)
        del features, edge_index, edge_weight, graph
        return new_graph


@LayerFactory.register('pyg_gin')
class GINLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "nn": nn.Sequential(nn.Linear(in_feat, out_feat), nn.BatchNorm1d(out_feat), nn.ReLU(), nn.Linear(out_feat, out_feat)),
            "train_eps": False,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.norm = normalization
        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = GINConv(**kwargs)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.activation(self.layer(features, edge_index))
        if self.norm:
            features = F.normalize(features)

        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_cheb')
class ChebLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "K": 2,
            "bias": True,
            "activation": "relu",
            "normalization": "sym"
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = ChebConv(in_channels=in_feat,
                              out_channels=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_attr if 'edge_attr' in graph else None

        features = self.activation(self.layer(
            features, edge_index, edge_weight))
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=edge_weight)
        del features, edge_index, edge_weight, graph

        return new_graph


class SAGELayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "root_weight": True,
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = SAGEConv(
            in_channels=in_feat, out_channels=out_feat, normalize=normalization, **kwargs)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.activation(self.layer(features, edge_index))
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_graphconv')
class GraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=None, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "aggr": "add",  # or "mean", "max"
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = GraphConv(in_channels=in_feat,
                               out_channels=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.activation(self.layer(features, edge_index))
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_gatedgraph')
class GatedGraphLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "num_layers": 1,
            "aggr": "add",  # or "mean", "max"
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = GatedGraphConv(
            out_channels=out_feat, num_layers=kwargs['num_layers'], **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.activation(self.layer(features, edge_index))
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_arma')
class ARMALayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "num_stacks": 1,
            "num_layers": 1,
            "shared_weights": False,
            "dropout": 0.0,
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = ARMAConv(in_channels=in_feat,
                              out_channels=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.activation(self.layer(features, edge_index))
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_graphunet')
class GraphUNetLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "depth": 3,
            "hidden_channels": out_feat,
            "pool_ratios": 0.5,
            "sum_res": True,
            "act": nn.ReLU(),
            "bias": True,
            "act": "relu",
            "activation": "relu"
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = GraphUNet(in_channels=in_feat,
                               out_channels=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index
        batch = graph.batch if 'batch' in graph else None

        features = self.layer(features, edge_index, batch=batch)
        features = self.activation(features)
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_agnn')
class AGNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "requires_grad": True,
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = AGNNConv(**kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.layer(features, edge_index)
        features = self.activation(features)
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph


@LayerFactory.register('pyg_appnp')
class APPNPLayer(nn.Module):
    def __init__(self, in_feat, out_feat, normalization=True, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            "K": 10,
            "alpha": 0.1,
            "bias": True,
            "activation": "relu",
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)

        self.activation = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else nn.Identity()
        self.layer = APPNP(
            in_channels=in_feat, out_channels=out_feat, normalize=normalization ** kwargs)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, graph: Data) -> Data:
        features = graph.x
        edge_index = graph.edge_index

        features = self.layer(features, edge_index)
        features = self.activation(features)
        features = self.dropout(features)

        new_graph = Data(x=features, edge_index=edge_index,
                         edge_attr=graph.edge_attr)
        del features, edge_index, edge_weight, graph

        return new_graph
