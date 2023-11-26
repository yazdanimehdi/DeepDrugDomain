from torch_geometric.data import Batch, Data
from dgl import DGLGraph, batch
from typing import Union, List


def batch_graphs(graphs: Union[List[Data], List[DGLGraph]]) -> Union[Batch, DGLGraph]:
    """
    Batch a list of graphs into a single graph.

    Parameters
    ----------
    graphs : Union[List[Data], List[DGLGraph]]
        A list of graphs to be batched.

    Returns
    -------
    Union[Batch, DGLGraph]
        A batched graph.
    """
    if isinstance(graphs[0], Data):
        return Batch.from_data_list(graphs)
    elif isinstance(graphs[0], DGLGraph):
        return batch(graphs)
    else:
        raise NotImplementedError(
            f"Batching for {type(graphs[0])} is not implemented.")
