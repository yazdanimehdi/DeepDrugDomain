from functools import partial
import warnings
from ..utils import LayerFactory, to_2tuple, ActivationFactory
from torch import nn


@LayerFactory.register('transformer_mlp')
class Mlp(nn.Module):
    """ 
    Multi-Layer Perceptron (MLP) used in Vision Transformer, MLP-Mixer, and related networks.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to the same as in_features.
        out_features (int, optional): Number of output features. Defaults to the same as in_features.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. If None, nn.Identity() is used. Defaults to None.
        bias (bool, optional): If set to True, layers will use bias. Defaults to True.
        drop (float, optional): Dropout rate. Defaults to 0.
        use_conv (bool, optional): Use Conv2D instead of Linear layer if True. Defaults to False.
    """

    def __init__(self, in_features, **kwargs):
        super().__init__()
        defaults = {
            "hidden_features": None,
            "out_features": None,
            "act_layer": "gelu",
            "norm_layer": None,
            "bias": True,
            "drop": 0.,
            "use_conv": False,
        }
        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        drop = kwargs["drop"]
        bias = kwargs["bias"]
        norm_layer = kwargs["norm_layer"]
        act_layer = ActivationFactory.create(kwargs["act_layer"])
        hidden_features = kwargs["hidden_features"]
        out_features = kwargs["out_features"]
        use_conv = kwargs["use_conv"]

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(
            nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = LayerFactory.create(
            norm_layer, hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
