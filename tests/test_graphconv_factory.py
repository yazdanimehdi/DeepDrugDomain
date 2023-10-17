import unittest
from deepdrugdomain.layers.utils.gcn.factory import AbstractGraphConvLayer, GraphConvLayerFactory


class TestGraphConvFactory(unittest.TestCase):

    def setUp(self):
        # Define a simple custom layer for testing purposes
        @GraphConvLayerFactory.register_layer('test_layer')
        class TestLayer(AbstractGraphConvLayer):
            def forward(self, *args, **kwargs):
                pass

    def test_register_layer(self):
        # Try to register an already registered layer type
        with self.assertRaises(ValueError):
            @GraphConvLayerFactory.register_layer('test_layer')
            class AnotherTestLayer(AbstractGraphConvLayer):
                def forward(self, *args, **kwargs):
                    pass

    def test_create_layer(self):
        # Create an instance of a registered layer
        layer_instance = GraphConvLayerFactory.create_layer('test_layer')
        self.assertIsInstance(layer_instance, AbstractGraphConvLayer)

        # Try to create an instance of a non-registered layer
        with self.assertRaises(ValueError):
            GraphConvLayerFactory.create_layer('nonexistent_layer')

    def tearDown(self):
        # Clean up (optional, in this case, we'll just remove our test layer from the registry)
        del GraphConvLayerFactory._registry['test_layer']


if __name__ == '__main__':
    unittest.main()
