import unittest
import torch
import dgl

from deepdrugdomain.utils.exceptions import MissingRequiredParameterError
from deepdrugdomain.layers import GraphConvLayerFactory  # Assuming layers.py contains your previous code


class TestGraphLayers(unittest.TestCase):

    def setUp(self):
        # Create a simple graph for testing
        self.g = dgl.graph(([0, 1, 2], [1, 2, 3]))
        self.features = torch.rand((4, 10))  # 4 nodes, 10-dimensional features

    def test_registered_layers(self):
        for layer_type in GraphConvLayerFactory._registry:
            with self.subTest(layer_type=layer_type):  # Using subTest to distinguish different layer tests
                layer = GraphConvLayerFactory.create_layer(layer_type, in_feat=10, out_feat=5)
                output = layer(self.g, self.features)

                # Adjust this part if layers have different output shapes
                self.assertEqual(output.shape, (4, 5))  # 4 nodes with 5-dimensional output each

    def test_missing_param_warning(self):
        for layer_type in GraphConvLayerFactory._registry:
            with self.subTest(layer_type=layer_type):
                with self.assertWarns(Warning):  # Ensure a warning is raised for a missing parameter
                    layer = GraphConvLayerFactory.create_layer(layer_type, in_feat=10, out_feat=5)

    def test_required_param_error(self):
        with self.assertRaises(
                MissingRequiredParameterError):  # Ensure an error is raised for a missing required parameter
            gat_layer = GraphConvLayerFactory.create_layer('dgl_gat', in_feat=10, out_feat=5)  # num_heads is missing


if __name__ == '__main__':
    unittest.main()
