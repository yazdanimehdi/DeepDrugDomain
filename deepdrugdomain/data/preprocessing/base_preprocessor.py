"""
Data Preprocessing Utility
--------------------------

This file contains classes and methods to handle preprocessing tasks for various data sources.
It provides mechanisms to preprocessing data, shard the processed data, and save and load the preprocessed data from disk.
Ray parallel processing is integrated to optimize the data processing speed.

Example:
    >>> from deepdrugdomain.data.preprocessing import BasePreprocessor
    >>> data = ["sample1", "sample2", "sample3"]
    >>> class MyPreprocessor(BasePreprocessor):
    ...     def __init__(self):
    ...         super(MyPreprocessor, self).__init__()
    ...     def preprocessing(self, data):
    ...         return data[::-1]  # For simplicity, we just reverse the string
    ...
    >>> preprocessor = MyPreprocessor()
    >>> info = preprocessor.process_and_get_info(data, "/path/to/save", in_memory=True, num_threads=2, file_prefix="test")
    >>> loaded_data = preprocessor.load_preprocessed_to_memory(preprocessor.get_saved_path("/path/to/save", "test"))

Classes:
    - AbstractBasePreprocessor: An abstract base class for all preprocessors. Sets the methods and interfaces to be implemented.
    - BasePreprocessor: A basic preprocessor implementation that provides utility methods and interfaces to deal with data preprocessing tasks.
      Note: This class is not intended to be instantiated directly.

Utility Functions:
    - save_mapping: Saves a given mapping into a JSON file.

Dependencies:
    - pickle: For data serialization and deserialization.
    - abc: Abstract base class definitions.
    - typing: For type hints.
    - os: Operating system interfaces, e.g., for file path operations.
    - json: To read and write JSON files.
    - ray: For parallel processing.
    - tqdm: A progress bar utility.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Optional
import os
import json
import ray
from tqdm import tqdm


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
        - data: The input data to preprocessing.

        Returns:
        - Processed data.
        """
        pass

    @abstractmethod
    def generate_mapping(self, data: Any) -> Dict[Any, Any]:
        """
        Abstract method to generate a mapping between original data and shards.

        Parameters:
        - data: The original data.

        Returns:
        - Dictionary mapping from data identifier to shard and position.
        """
        pass

    @abstractmethod
    def process_and_get_info(self, data: Any, attribute: str, save_directory: str, in_memory: bool,
                             *args, **kwargs) -> Dict[str, Any]:
        """
            Process and store the input data, then provide details such as the preprocessed data, shard metadata, and mapping indices.

            Parameters:
                data (Any): The raw input data intended for preprocessing.
                attribute (str): Specifies which attribute of the input data this preprocessor is responsible for.
                save_directory (str): Path to the directory where the preprocessed data files will be stored.
                in_memory (bool, optional): A flag to determine whether the preprocessed data should be retained in memory or stored as individual files on disk. Defaults to `False`.

            Returns:
                dict: A dictionary containing:
                    - "preprocessed_data" (Any): The processed data.
                    - "mapping" (dict): A mapping between the original data entries and their corresponding shard and position details.
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


class BasePreprocessor(AbstractBasePreprocessor, ABC):
    """
        Base class for preprocessing data.

        This class offers essential mechanisms to preprocessing data and either store it in memory or save it
        individually to files using the pickle format. When data is kept in memory, it can be saved to disk for
        future use without reprocessing.

        It's recommended to inherit from this class to harness the foundational preprocessing functions.
        For more tailored preprocessing logic, users can override the methods of this class or inherit
        from `AbstractBasePreprocessor` and craft the required methods from the outset.

        Direct instantiation of this class is restricted.
    """
    def __init__(self, **kwargs):
        """
           Initialize the BasePreprocessor.

           Parameters:
               kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.kwargs = kwargs
        self.none = []

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
           Process the data. Needs to be overridden in derived classes.

           Parameters:
               data (Any): The data to preprocessing.

           Returns:
               Any: The preprocessed data.
        """
        pass

    @ray.remote
    def _worker(self, data):
        """
           Worker function to preprocessing data in a distributed manner using Ray.

           Parameters:
               data (Any): The data to preprocessing.

           Returns:
               Any: The preprocessed data.
        """
        # This is just a stub, the actual implementation would be in derived classes or can be overridden.
        return self.preprocess(data)

    @ray.remote
    def _preprocess_and_save_data_point(self, data_point: Any, base_dir: str, idx: int, prefix: str) -> Optional[str]:
        """
           Preprocess a single data point and save it to disk using pickle.

           Parameters:
               data_point (Any): The data point to preprocessing.
               base_dir (str): Directory to save the preprocessed data.
               idx (int): Index of the data point.
               prefix (str): Prefix for the filename.

           Returns:
               str or None: Path to the saved file or None if saving or preprocessing failed.
        """
        processed = self.preprocess(data_point)
        registered_name = self.__class__.__name__
        if processed is not None:
            # Assuming your save function accepts the base directory as a parameter
            path = os.path.join(base_dir, f"{registered_name}_{prefix}_object_{idx}.bin")
            self.save_data(processed, path)
            return path

        return None

    def _process(self, futures, data):
        """
            Process data items in parallel using Ray.

            Parameters:
                futures (dict): Dictionary of Ray futures.
                data (List[Any]): List of data items to process.

            Returns:
                tuple: A tuple containing all processed data and invalid results.
        """
        invalid_results = []
        all_processed_data = {}
        for data_item in tqdm(futures, total=len(data), desc=f"Processing"):
            processed = ray.get(futures[data_item])
            all_processed_data[data_item] = processed
            if processed is None:  # 2. Check if the result is None
                invalid_results.append(data_item)

        return all_processed_data, invalid_results

    def process_and_get_info(self, data: List[Any], directory: str, in_memory: bool = True,
                             num_threads: int = 4, file_prefix: str = "", *args, **kwargs) -> Dict[Any, Any]:
        """
           Process data and retrieve relevant information.

           Parameters:
               data (List[Any]): Data items to process.
               directory (str): Directory to save preprocessed data and mapping info.
               in_memory (bool, optional): Whether to keep preprocessed data in memory or save to disk.
               num_threads (int, optional): Number of threads for parallel processing using Ray.
               file_prefix (str, optional): Prefix for saved filenames.
               *args, **kwargs: Additional arguments.

           Returns:
               dict: Dictionary containing mapping info and processed data.

        """

        ray.init(num_cpus=num_threads)
        registered_name = self.__class__.__name__
        if in_memory:
            futures = {d: self._worker.remote(self, d) for d in data}
            all_processed_data, invalid_results = self._process(futures, data)
            self.none = invalid_results
            mapping_info = self.generate_mapping(data)
            self.save_preprocessed_to_disk(all_processed_data, directory, file_prefix)
            save_mapping(mapping_info, os.path.join(directory, f"{registered_name}_{file_prefix}_mapping_info.json"))
            ray.shutdown()
            # Return the mapping info and the processed data
            return {
                'mapping_info': mapping_info,
                'processed_data': all_processed_data
            }

        else:
            futures = {d: self._preprocess_and_save_data_point.remote(self, d, directory, idx, file_prefix) for idx, d in enumerate(data)}
            all_processed_data, invalid_results = self._process(futures, data)
            self.none = invalid_results
            save_mapping(all_processed_data,
                         os.path.join(directory, f"{registered_name}_{file_prefix}_mapping_info.json"))
            ray.shutdown()
            return {'mapping_info': all_processed_data}

    def generate_mapping(self, data: List[Any]) -> Dict[Any, Any]:
        """
            Generates a mapping between data items and their position.

            Parameters:
            - data (List[Any]): List of data items.

            Returns:
            - Dict[Any, Any]: A dictionary mapping each data item to its position,
                              setting it to None if it's in the list of unprocessed data.
        """
        mapping = {}

        for idx, item in enumerate(data):
            if item in self.none:
                mapping[item] = None
            else:
                mapping[item] = 0

        return mapping

    def collect_invalid_data(self, online: bool, data: Optional[List[str]] = None) -> List[str]:
        """
            Identify and collect data points that result in None after preprocessing.

            Parameters:
            - online (bool): If True, the data is processed and checked; if False,
                             the current list of invalid data is returned.
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
        """

        self.none = invalid_data

    def _serialize_key(self, data: Any) -> Any:
        """
        Convert the data (key) to a format suitable for saving.
        For simplicity and due to the use of pickle as the default save mechanism,
        we don't need to serialize the data, so the data itself is returned.

        Parameters:
        - data: The data to be serialized.

        Returns:
        - Serialized data.
        """
        return data

    def _serialize_value(self, data: Any) -> Any:
        """
        Convert the data (value) to a format suitable for saving.
        For simplicity and due to the use of pickle as the default save mechanism,
        we don't need to serialize the data, so the data itself is returned.

        Parameters:
        - data: The data to be serialized.

        Returns:
        - Serialized data.
        """
        return data

    def _deserialize_key(self, data: Any) -> Any:
        """
        Convert the saved data key back to its original format.
        As we're using pickle and not performing any special serialization,
        the original data is directly returned.

        Parameters:
        - data: The data key to be deserialized.

        Returns:
        - Deserialized data key.
        """
        return data

    def _deserialize_value(self, data: Any) -> Any:
        """
        Convert the saved data value back to its original format.
        As we're using pickle and not performing any special serialization,
        the original data is directly returned.

        Parameters:
        - data: The data value to be deserialized.

        Returns:
        - Deserialized data value.
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
        """
           Save all the preprocessed data to a single file using pickle.

           Parameters:
           - data (Dict[Any, Any]): The preprocessed data.
           - path: The directory where the data should be saved.
           - file_prefix: Prefix to be added to the filename.
        """
        registered_name = self.__class__.__name__
        file_name = f"{registered_name}_{file_prefix}_all.pkl"

        serialized_data = {self._serialize_key(key): self._serialize_value(value) for key, value in data.items()}
        with open(os.path.join(path, file_name), 'wb') as fp:
            pickle.dump(serialized_data, fp)

    def load_preprocessed_to_memory(self, path: str) -> Any:
        """
            Load all the preprocessed data from a file.

            Parameters:
            - path: Path to the preprocessed data file.

            Returns:
            - Dict[Any, Any]: The loaded and deserialized preprocessed data.
        """

        with open(path, "rb") as fp:
            data = pickle.load(fp)

        original_data = {self._deserialize_key(key): self._deserialize_value(value) for key, value in data.items()}
        return original_data

    def get_saved_path(self, path: str, prefix: str) -> Any:
        """
            Construct the path of the saved preprocessed data file.

            Parameters:
            - path: Base directory of the saved file.
            - prefix: Prefix added to the filename.

            Returns:
            - str: Full path to the saved file.

        """
        registered_name = self.__class__.__name__
        file_name = f"{registered_name}_{prefix}_all.pkl"

        return os.path.join(path, file_name)
