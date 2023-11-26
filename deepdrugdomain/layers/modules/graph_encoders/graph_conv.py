from ...utils import LayerFactory
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Dict, Any, Optional, Union


class GraphConvEncoder(nn.Module):
    """
        Graph convolutional encoder module.
    """

    def __init__(self, conv_layer_types: Union[Sequence[str], str], input_size: int, output_size: int, hidden_sizes: Sequence[int], pooling_layer_type: Optional[str], pooling_layer_kwargs: Optional[Dict[str, Any]] = {}, conv_layer_kwargs: Optional[Sequence[Dict[str, Any]]] = None, dropout: Optional[Sequence[float]] = None, normalization: Optional[Sequence[bool]] = None,  **kwargs) -> None:
        """
            Initializes the graph convolutional encoder module.
            args:
                conv_layer_types (Sequence[str]): The types of graph convolution layers to use.
                input_size (int): The input size of the module.
                output_size (int): The output size of the module.
                hidden_sizes (Sequence[int]): The hidden sizes of the module.
                pooling_layer_type (Optional[str]): The type of pooling layer to use.
                pooling_layer_kwargs (Optional[Dict[str, Any]]): The keyword arguments for the pooling layer.
                dropout (Sequence[float]): The dropout rates for the graph convolution layers.
                normalization (Sequence[bool]): The normalization types for the graph convolution layers.
                **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.kwargs = kwargs
        dims = [input_size] + list(hidden_sizes) + [output_size]
        conv_kwargs = conv_layer_kwargs if conv_layer_kwargs else [
            {}] * (len(dims) - 1)
        normalization = normalization if normalization else [
            False] * (len(dims) - 1)
        dropout = dropout if dropout else [0.0] * (len(dims) - 1)

        assert len(dims) - 1 == len(dropout) == len(conv_kwargs) == len(normalization) == len(
            conv_layer_types), "The number of graph convolution layers parameters must be the same"

        self.graph_conv = nn.ModuleList([
            LayerFactory.create(conv_layer_types[i],
                                dims[i],
                                dims[i + 1],
                                normalization=normalization[i],
                                dropout=dropout[i],
                                **conv_kwargs[i]) for i in range(len(dims) - 1)])

        self.pooling = LayerFactory.create(
            pooling_layer_type, **pooling_layer_kwargs) if pooling_layer_type else None

    def forward(self, graph: Any) -> Any:
        """
            Forward pass of the module.
            args:
                graph (Any): The input graph.
            returns:
                Any: The output graph or encoded Tensor.
        """
        # todo: create a data type for graphs that can be modified easily if needed

        for layer in self.graph_conv:
            graph = layer(graph)

        if self.pooling:
            rep = self.pooling(graph)
            return rep

        return graph
