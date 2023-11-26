import os
import json
import warnings
from typing import Any, Dict, List, Optional
from deepdrugdomain.data import DatasetFactory
from deepdrugdomain.data.collate import CollateFactory
from deepdrugdomain.models.factory import ModelFactory
from torch.utils.data.dataloader import default_collate


def read_config(config_path: str) -> Dict[str, Any]:
    """
    Reads a JSON configuration file and returns the configuration as a dictionary.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary with settings for the model and dataset.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def initialize_training_environment(model_name: str, dataset_name: str, split_ratios: List[float], config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None) -> tuple:
    """
    Initializes the training environment by loading the model, dataset, and collate function configurations.
    It either uses a provided configuration dictionary or loads a configuration from a file. If neither is provided,
    it defaults to a configuration file named '{model_name}.json' in the 'deepdrugdomain/configs' directory.

    Args:
        model_name (str): The name of the model to load.
        dataset_name (str): The name of the dataset to load.
        split_ratios (List[float]): The ratios to split the dataset into training, validation, and test sets.
        config (Optional[Dict[str, Any]]): An optional pre-loaded configuration dictionary.
        config_path (Optional[str]): An optional path to the configuration file.

    Returns:
        tuple: A tuple containing the model, dataset splits, and collate function.

    Raises:
        AssertionError: If certain keys are not present in the configuration or the specified dataset does not match.
        FileNotFoundError: If the configuration file does not exist.
    """
    default_config_path = 'deepdrugdomain/configs'
    default_config = False

    # Load configuration from file if not provided
    if not config:
        if not config_path:
            warnings.warn(
                "Using default configuration file located in 'deepdrugdomain/configs'", UserWarning)
            config_path = os.path.join(
                default_config_path, f"{model_name}.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file for model '{model_name}' not found at {config_path}.")
            default_config = True
        config = read_config(config_path)

    # Validate configuration
    assert 'model' in config, "Configuration file must contain a 'model' key"
    assert 'dataset' in config, "Configuration file must contain a 'dataset' key"
    assert dataset_name in config['dataset'], f"Dataset '{dataset_name}' does not match any of the datasets specified in the configuration file."

    # Create dataset and splits
    dataset_config = config['dataset'][dataset_name]
    dataset = DatasetFactory.create(dataset_name, **dataset_config)
    dataset_splits = dataset.split(split_ratios)

    # Determine if a custom collate function is specified and create it, else use default collate
    collate_config = dataset_config.get('collate_fn')
    if collate_config:
        collate_fn_name = collate_config['name']
        collate_fn_args = collate_config.get('args', {})
        collate_fn = CollateFactory.create(collate_fn_name, **collate_fn_args)
    else:
        collate_fn = default_collate
        warnings.warn(
            "No custom 'collate_fn' specified, using default collate function.", UserWarning)

    # Determine model version and create model
    model_version = dataset_config.get('model_version', "default")
    if default_config:
        model = ModelFactory.create(model_name, version=model_version)
    else:
        model = ModelFactory.create(model_name, **config['model'])

    return model, dataset_splits, collate_fn
