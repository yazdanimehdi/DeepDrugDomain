import dgl
from torch import Tensor
import torch_geometric
from typing import Union


def dgl_to_pyg(g: dgl.DGLGraph) -> torch_geometric.data.Data:
    """
    Convert a DGL graph to a PyG data object.

    :param g: The DGL graph to convert.
    :return: The PyG data object.
    """
    return torch_geometric.data.Data(x=g.ndata['h'], edge_index=g.edges()[0], edge_attr=g.edata['w'])


def pyg_to_dgl(data: torch_geometric.data.Data) -> dgl.DGLGraph:
    """
    Convert a PyG data object to a DGL graph.

    :param data: The PyG data object to convert.
    :return: The DGL graph.
    """
    g = dgl.graph((data.edge_index[0], data.edge_index[1]))
    g.ndata['h'] = data.x
    g.edata['w'] = data.edge_attr

    return g


def change_node_attr(graph: Union[dgl.DGLGraph, torch_geometric.data.Data], new_attr: Tensor) -> Union[dgl.DGLGraph, torch_geometric.data.Data]:
    """
    Change the attributes of a graph.

    :param graph: The graph to change the attributes of.
    :param new_attr: The new attributes.
    :return: The graph with the changed attributes.
    """
    if isinstance(graph, dgl.DGLGraph):
        graph.ndata['h'] = new_attr
    elif isinstance(graph, torch_geometric.data.Data):
        graph.x = new_attr
    else:
        raise ValueError(f"Unknown graph type '{type(graph)}'.")

    return graph


def change_edge_attr(graph: Union[dgl.DGLGraph, torch_geometric.data.Data], new_edge_attr: Tensor) -> Union[dgl.DGLGraph, torch_geometric.data.Data]:
    """
    Change the edge attributes of a graph.

    :param graph: The graph to change the edge attributes of.
    :param new_edge_attr: The new edge attributes.
    :return: The graph with the changed edge attributes.
    """
    if isinstance(graph, dgl.DGLGraph):
        graph.edata['w'] = new_edge_attr
    elif isinstance(graph, torch_geometric.data.Data):
        graph.edge_attr = new_edge_attr
    else:
        raise ValueError(f"Unknown graph type '{type(graph)}'.")

    return graph


def get_node_attr(graph: Union[dgl.DGLGraph, torch_geometric.data.Data]) -> Tensor:
    """
    Get the node attributes of a graph.

    :param graph: The graph to get the node attributes of.
    :return: The node attributes.
    """
    if isinstance(graph, dgl.DGLGraph):
        return graph.ndata['h']
    elif isinstance(graph, torch_geometric.data.Data):
        return graph.x
    else:
        raise ValueError(f"Unknown graph type '{type(graph)}'.")


def get_edge_attr(graph: Union[dgl.DGLGraph, torch_geometric.data.Data]) -> Tensor:
    """
    Get the edge attributes of a graph.

    :param graph: The graph to get the edge attributes of.
    :return: The edge attributes.
    """
    if isinstance(graph, dgl.DGLGraph):
        return graph.edata['w']
    elif isinstance(graph, torch_geometric.data.Data):
        return graph.edge_attr
    else:
        raise ValueError(f"Unknown graph type '{type(graph)}'.")
