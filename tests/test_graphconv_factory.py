import unittest
from deepdrugdomain.layers.graph_layers import *
import torch
from dgl import DGLGraph


class TestFactoryFunctionality(unittest.TestCase):

    def test_register_graph_conv_layer(self):
        # Attempt to register a new layer
        @GraphConvLayerFactory.register_layer('test_conv_layer')
        class TestGraphConvLayer(AbstractGraphConvLayer):
            pass

        # Ensure the layer has been added to the registry
        self.assertIn('test_conv_layer', GraphConvLayerFactory._registry)

    def test_register_graph_pooling_layer(self):
        # Attempt to register a new layer
        @GraphPoolingLayerFactory.register_layer('test_pool_layer')
        class TestGraphPoolingLayer(AbstractGraphPoolingLayer):
            pass

        # Ensure the layer has been added to the registry
        self.assertIn('test_pool_layer', GraphPoolingLayerFactory._registry)

    def test_duplicate_registration_error(self):
        # Attempt to register the same layer type twice to ensure an error is raised
        with self.assertRaises(ValueError):
            @GraphConvLayerFactory.register_layer('test_conv_layer')
            class AnotherTestGraphConvLayer(AbstractGraphConvLayer):
                pass

    def test_unknown_layer_creation_error(self):
        # Try to create an unknown layer type to ensure an error is raised
        with self.assertRaises(ValueError):
            GraphConvLayerFactory.create_layer('non_existent_layer_type')

    def test_layer_creation(self):
        # Register and create a layer to ensure proper functionality
        @GraphPoolingLayerFactory.register_layer('creation_test_layer')
        class CreationTestLayer(AbstractGraphPoolingLayer):
            pass

        layer_instance = GraphPoolingLayerFactory.create_layer('creation_test_layer')
        self.assertIsInstance(layer_instance, CreationTestLayer)


class TestFactory(unittest.TestCase):

    # Test Graph Convolution Layers
    def test_graph_conv_layers(self):
        # Mock data
        g = DGLGraph()
        g.add_nodes(10)
        g.add_edges(list(range(1, 10)), 0)
        features = torch.randn(10, 3)

        # Ensure that each registered graph convolution layer can be instantiated and called
        for layer_type in GraphConvLayerFactory._registry:
            layer = GraphConvLayerFactory.create_layer(layer_type, 3, 3)
            self.assertIsInstance(layer, AbstractGraphConvLayer)  # Ensure correct inheritance
            output = layer(g, features)
            self.assertIsInstance(output, torch.Tensor)  # Ensure correct output type

    # Test Graph Pooling Layers
    def test_graph_pooling_layers(self):
        # Mock data
        g = DGLGraph()
        g.add_nodes(10)
        g.add_edges(list(range(1, 10)), 0)
        features = torch.randn(10, 3)

        # Ensure that each registered graph pooling layer can be instantiated and called
        for layer_type in GraphPoolingLayerFactory._registry:
            layer = GraphPoolingLayerFactory.create_layer(layer_type, 3, 3)
            self.assertIsInstance(layer, AbstractGraphPoolingLayer)  # Ensure correct inheritance
            output = layer(g, features)
            self.assertIsInstance(output, torch.Tensor)  # Ensure correct output type


if __name__ == "__main__":
    unittest.main()
