import networkx as nx
import dgl


def serialize_dgl_graph_hdf5(graph: dgl.DGLGraph) -> dict:
    """
    Convert a DGL graph into a dictionary for easier serialization.

    This utility function is designed to take a DGL graph object and transform
    it into a dictionary format. This representation primarily includes the graph's
    adjacency matrix. If present, it also includes node and edge features,
    as well as batch information which can be useful for batched graph processing.

    Parameters:
    - graph (dgl.DGLGraph): The DGL graph object to be serialized.

    Returns:
    - dict: A dictionary containing the graph's serialized components including adjacency matrix,
      node features, edge features, and batch information.
    """
    netx_graph = dgl.to_networkx(graph)
    adjacency = nx.to_numpy_array(netx_graph)

    serialized_graph = {
        'adjacency': adjacency
    }

    # Include batch information if present
    if hasattr(graph, 'batch_num_nodes'):
        serialized_graph['batch_num_nodes'] = graph.batch_num_nodes()
    if hasattr(graph, 'batch_num_edges'):
        serialized_graph['batch_num_edges'] = graph.batch_num_edges()

    # Include node and edge features if present
    if graph.ndata:
        serialized_graph['node_features'] = graph.ndata
    if graph.edata:
        serialized_graph['edge_features'] = graph.edata

    return serialized_graph


def deserialize_dgl_graph_hdf5(serialized_graph: dict) -> dgl.DGLGraph:
    """
    Convert a serialized graph dictionary back into a DGL graph.

    This utility function takes a dictionary representation of a DGL graph,
    which typically contains an adjacency matrix, node features, edge features, and
    batch information. It then reconstructs the DGL graph object from these components.

    Parameters:
    - serialized_graph (dict): A dictionary containing the graph's serialized components.

    Returns:
    - dgl.DGLGraph: The reconstructed DGL graph object.
    """
    netx_graph = nx.from_numpy_array(serialized_graph['adjacency'])
    graph = dgl.from_networkx(netx_graph)

    # Recover node and edge features if present
    if 'node_features' in serialized_graph:
        graph.ndata.update(serialized_graph['node_features'])
    if 'edge_features' in serialized_graph:
        graph.edata.update(serialized_graph['edge_features'])

    # Recover batch information if present
    if 'batch_num_nodes' in serialized_graph:
        graph.batch_num_nodes = serialized_graph['batch_num_nodes']
    if 'batch_num_edges' in serialized_graph:
        graph.batch_num_edges = serialized_graph['batch_num_edges']

    return graph
