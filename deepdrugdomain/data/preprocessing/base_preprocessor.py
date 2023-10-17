"""
base_preprocessor.py

Provides an abstract class for defining preprocessing operations for various data types and a base implementation
that processes data, shards it, and saves it using the HDF5 format.
"""
import multiprocessing
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Optional
import h5py
import os
from multiprocessing import Pool, Manager
import json

import ray
from tqdm import tqdm
import logging


class AbstractBasePreprocessor(ABC):
    """
    An abstract base class for data preprocessors.
    All custom preprocessors should inherit from this class.
    """

    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
        Abstract method for preprocessing the given data.

        Parameters:
        - data: The input data to preprocess.

        Returns:
        - Processed data.
        """
        pass

    @abstractmethod
    def generate_mapping(self, data: Any, shard_size: Union[int, None]) -> Dict[Any, Any]:
        """
        Abstract method to generate a mapping between original data and shards.

        Parameters:
        - data: The original data.
        - shard_size: The size of each shard. If None, data isn't sharded.

        Returns:
        - Dictionary mapping from data identifier to shard and position.
        """
        pass

    @abstractmethod
    def process_and_get_info(self, data: Any, attribute: str, shard_directory: str, shard_size: Union[int, None],
                             *args, **kwargs) -> Dict[str, Any]:
        """
        Process the input data and return all relevant details,
        including preprocessed data, shard information, and mappings.

        Parameters:
        - data: The input data to preprocess.
        - shard_directory: The directory where shards should be saved.
        - shard_size: The size of each shard. If None, data isn't sharded.

        Returns:
        - Dictionary with keys:
          - "preprocessed_data": the processed data
          - "shard_info": shard-related details (e.g., file paths)
          - "mapping": dictionary mapping original data to shard and position.
        """
        pass

    @abstractmethod
    def _serialize_key(self, data: Any) -> Any:
        """
        Convert the data (key) to a format that can be saved in format of choice.

        Parameters:
        - data: The data to be serialized.

        Returns:
        - Serialized data.
        """
        pass

    @abstractmethod
    def _serialize_value(self, data: Any) -> Any:
        """
        Convert the data (value) to a format that can be saved in format of choice.

        Parameters:
        - data: The data to be serialized.

        Returns:
        - Serialized data.
        """
        pass

    @abstractmethod
    def _deserialize_key(self, data: Any) -> Any:
        """
        Convert the data (key) back to its original format.

        Parameters:
        - data: The data to be deserialized.

        Returns:
        - Deserialized data.
        """
        pass

    @abstractmethod
    def _deserialize_value(self, data: Any) -> Any:
        """
        Convert the data (value) back to its original format.

        Parameters:
        - data: The data to be deserialized.

        Returns:
        - Deserialized data.
        """
        pass

    @abstractmethod
    def save_data(self, data: Any, path: str) -> None:
        """
        Abstract method to save the preprocessed data.

        Parameters:
        - data: The data to save.
        - path: Path where the data should be saved.

        Returns:
        - None
        """
        pass

    @abstractmethod
    def load_data(self, path: str) -> Any:
        """
        Abstract method to load the preprocessed data.

        Parameters:
        - path: Path from where the data should be loaded.

        Returns:
        - Loaded data.
        """
        pass

    @abstractmethod
    def save_preprocessed_to_disk(self, data: Dict[Any, Any], path: str, file_prefix: str) -> None:
        """
        Serialize and save the preprocessed data to disk.

        Parameters:
        - data (Dict[Any, Any]): The preprocessed data to be saved.
        - path (str): The directory where the data should be saved.
        - file_prefix (str): Prefix to be added to the file name.

        Note:
        The resulting file will be named as {class_name}_{file_prefix}_all.pkl
        """
        pass

    @abstractmethod
    def load_preprocessed_to_memory(self, path: str) -> Any:
        """
        Load and deserialize the preprocessed data from disk into memory.

        Parameters:
        - path (str): The path of the file containing the preprocessed data.

        Returns:
        - Any: The deserialized data.

        Note:
        The data should have been saved using `save_preprocessed_to_disk` method.
        """
        pass


def save_mapping(mapping: dict, filename: str) -> None:
    """
    Save a mapping to a JSON file.

    Parameters:
    - mapping (dict): The mapping data to be saved.
    - filename (str): The name of the file where the mapping data will be saved.

    Returns:
    - None
    """
    with open(filename, 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)


class BasePreprocessor(AbstractBasePreprocessor):
    """
    An example implementation of the AbstractBasePreprocessor.
    Provides basic mechanisms to preprocess data, shard it, and save it using the HDF5 format.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.none = []

    def preprocess(self, data: Any) -> Any:
        raise NotImplementedError

    @ray.remote
    def _worker(self, data):
        # This is just a stub, the actual implementation would be in derived classes or can be overridden.
        return self.preprocess(data)

    @ray.remote
    def _preprocess_and_save_data_point(self, data_point: Any, base_dir: str, idx: int, prefix: str) -> Optional[str]:

        processed = self.preprocess(data_point)
        registered_name = self.__class__.__name__
        if processed is not None:
            # Assuming your save function accepts the base directory as a parameter
            path = os.path.join(base_dir, f"{registered_name}_{prefix}_object_{idx}.bin")
            self.save_data(processed, path)
            return path

        return None

    def _process(self, futures, data):
        invalid_results = []
        all_processed_data = {}
        for data_item in tqdm(futures, total=len(data), desc=f"Processing"):
            processed = ray.get(futures[data_item])
            all_processed_data[data_item] = processed
            if processed is None:  # 2. Check if the result is None
                invalid_results.append(data_item)

        ray.shutdown()
        return all_processed_data, invalid_results

    def process_and_get_info(self, data: List[Any], directory: str, in_memory: bool = True,
                             num_threads: int = 4, file_prefix: str = "", *args, **kwargs) -> Dict[Any, Any]:
        ray.init(num_cpus=num_threads)
        registered_name = self.__class__.__name__
        if in_memory:
            futures = {d: self._worker.remote(self, d) for d in data}
            all_processed_data, invalid_results = self._process(futures, data)
            self.none = invalid_results
            mapping_info = self.generate_mapping(data, None)
            self.save_preprocessed_to_disk(all_processed_data, directory, file_prefix)
            save_mapping(mapping_info, os.path.join(directory, f"{registered_name}_{file_prefix}_mapping_info.json"))
            ray.shutdown()
            # Return the mapping info and the processed data
            return {
                'mapping_info': mapping_info,
                'processed_data': all_processed_data
            }

        else:
            futures = {d: self._preprocess_and_save_data_point.remote(self, d, directory) for d in data}
            all_processed_data, invalid_results = self._process(futures, data)
            self.none = invalid_results
            save_mapping(all_processed_data, os.path.join(directory, f"{registered_name}_{file_prefix}_mapping_info.json"))
            ray.shutdown()
            return {'mapping_info': all_processed_data}

    def generate_mapping(self, data: List[Any], shard_size: Union[int, None]) -> Dict[Any, Any]:
        """
        Generates a mapping between data items and their shard & index.

        Parameters:
        - data (List[Any]): List of data items.
        - shard_size (Union[int, None]): The number of data items per shard.

        Returns:
        - Dict[Any, Any]: A dictionary mapping each data item to its shard and index.
        """
        mapping = {}
        shard_id = 0
        for idx, item in enumerate(data):
            if item in self.none:
                mapping[item] = None
            else:
                if shard_size and idx % shard_size == 0 and idx > 0:
                    shard_id += 1
                mapping[item] = shard_id
        return mapping

    def collect_invalid_data(self, online: bool, data: Optional[List[str]] = None) -> List[str]:
        """
        Identify and collect data points that result in None after preprocessing.

        Parameters:
        - online (bool): If True, the data is processed and checked; if False, the current list of invalid data is returned.
        - data (Optional[List[str]]): List of data to be checked. Only used if online is True.

        Returns:
        - List[str]: List of data points that are considered invalid or unprocessed.
        """
        if online and data:
            for d in data:
                if self.preprocess(d) is None:
                    self.none.append(d)

        return self.none

    def set_invalid_data(self, invalid_data: List[str]) -> None:
        """
        Set a list of data points that are considered invalid or unprocessed.

        Parameters:
        - invalid_data (List[str]): List of data points to set as invalid.

        Returns:
        - None
        """
        self.none = invalid_data

    def _serialize_key(self, data: Any) -> Any:
        """
        Convert the data (key or value) to a format that can be saved in HDF5.

        Parameters:
        - data: The data to be serialized.

        Returns:
        - Serialized data.
        """
        return data

    def _serialize_value(self, data: Any) -> Any:
        """
        Convert the data (value) to a format that can be saved in HDF5.

        Parameters:
        - data: The data to be serialized.

        Returns:
        - Serialized data.
        """
        return data

    def _deserialize_key(self, data: Any) -> Any:
        """
        Convert the data (key) back to its original format.

        Parameters:
        - data: The data to be deserialized.

        Returns:
        - Deserialized data.
        """
        return data

    def _deserialize_value(self, data: Any) -> Any:
        """
        Convert the data (value) back to its original format.

        Parameters:
        - data: The data to be deserialized.

        Returns:
        - Deserialized data.
        """
        return data

    def save_data(self, data: Any, path: str) -> None:
        """
        Save the data using pickle.

        Parameters:
        - data: The data to save.
        - path: Path where the data should be saved.

        Returns:
        - None
        """
        with open(path, 'wb') as fp:
            pickle.dump(data, fp)

    def load_data(self, path: str) -> Any:
        """
        Load the data using pickle.

        Parameters:
        - path: Path from where the data should be loaded.

        Returns:
        - Loaded data.
        """
        with open(path, 'rb') as fp:
            loaded_data = pickle.load(fp)
        return loaded_data

    def save_preprocessed_to_disk(self, data: Dict[Any, Any], path: str, file_prefix: str) -> None:

        registered_name = self.__class__.__name__
        file_name = f"{registered_name}_{file_prefix}_all.pkl"

        serialized_data = {self._serialize_key(key): self._serialize_value(value) for key, value in data.items()}
        with open(os.path.join(path, file_name), 'wb') as fp:
            pickle.dump(serialized_data, fp)

    def load_preprocessed_to_memory(self, path: str) -> Any:

        with open(path, "rb") as fp:
            data = pickle.load(fp)

        original_data = {self._deserialize_key(key): self._deserialize_value(value) for key, value in data.items()}
        return original_data

    def get_saved_path(self, path: str, prefix: str) -> Any:
        registered_name = self.__class__.__name__
        file_name = f"{registered_name}_{prefix}_all.pkl"

        return os.path.join(path, file_name)