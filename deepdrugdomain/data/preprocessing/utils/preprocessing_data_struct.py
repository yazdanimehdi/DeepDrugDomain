import json
import os
from typing import Dict, Any, List, Tuple
from .. import PreprocessorFactory, BasePreprocessor
from typing import overload, TypeVar, Dict, Union
from collections import OrderedDict
from abc import ABC, abstractmethod


class PreprocessingObjectBase(ABC):

    @abstractmethod
    def __init__(self, attribute: str, from_dtype: str, to_dtype: str, preprocessing_settings: Dict[str, Any], in_memory: bool = True, online: bool = False) -> None:
        """
        Abstract base class for preprocessing objects.

        This class serves as a base for defining preprocessing tasks which include
        data type transformations and specific preprocessing settings. It supports
        both in-memory and online modes for data preprocessing.

        Attributes:
        -----------
        attribute: str
            The name of the attribute to be preprocessed.
        from_dtype: str
            The data type before preprocessing.
        to_dtype: str
            The data type after preprocessing.
        preprocessing_settings: Dict[str, Any]
            A dictionary containing settings for the preprocessing.
        in_memory: bool, default=True
            Flag to indicate if preprocessing should be done in-memory.
        online: bool, default=False
            Flag to indicate if preprocessing should be done online.
        preprocess: BasePreprocessor
            The preprocessor object created based on the provided data types.
        preprocessing_type: str
            The name of the preprocessing type.

        Methods:
        --------
        __init__(self, attribute, from_dtype, to_dtype, preprocessing_settings, in_memory, online)
            Initializes the preprocessing object with the given attributes and settings.
        _get_unique_data(self, data)
            Get the unique data from the attribute(s).
        _preprocessing_done(self, directory)
            Check if preprocessing has been done for the entire dataset.
        _clean_data(self, data, data_unique)
            Clean the data by removing rows where processed data is None for
            any of the preprocessed attributes.
        _load_mapping(self, directory)
            Load the mapping dictionary from the shard directory.
        __call__(self, data, directory, threads)
            Perform preprocessing on the data.
        __repr__(self)
            Return a string representation of the preprocessing object.
        __add__(self, other)
            Add two preprocessing objects together.
        """
        if in_memory is None:
            in_memory = True
        if online is None:
            online = True

        self.attribute = attribute
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        self.preprocessing_settings = preprocessing_settings
        self.in_memory = in_memory
        self.online = online
        self.preprocess = PreprocessorFactory.create(
            from_dtype, to_dtype, None, **preprocessing_settings)
        self.preprocessing_type = PreprocessorFactory.get_preprocessor_name(
            from_dtype, to_dtype)

    def _get_unique_data(self, data) -> List[Any]:
        """
        Get the unique data from the attribute(s).

        Returns:
        - List[Any]: The unique data from the attribute(s).
        """
        if isinstance(self.attribute, list):
            data_unique = data[self.attribute].drop_duplicates().tolist()
        else:
            data_unique = data[self.attribute].unique().tolist()

        return data_unique

    def _preprocessing_done(self, directory) -> bool:
        """
        Check if preprocessing has been done for the entire dataset.

        Returns:
        - bool: True if preprocessing files are found, False otherwise.
        """
        mapping_path = os.path.join(
            directory, f"{self.preprocessing_type}_{self.attribute}_mapping_info.json")

        return os.path.exists(mapping_path)

    def _clean_data(self, data, data_unique) -> None:
        """
        Clean the data by removing rows where processed data is None for
        any of the preprocessed attributes.
        """
        none_col_list = self.preprocess.collect_invalid_data(
            self.online, data_unique)

        return data[~data[self.attribute].isin(none_col_list)]

    def _load_mapping(self, directory) -> Dict[Any, Any]:
        """
        Load the mapping dictionary from the shard directory.

        Returns:
        - Dict: mapping dictionary of the relevant data
        """
        mapping_path = os.path.join(directory,
                                    f"{self.preprocessing_type}_{self.attribute}_mapping_info.json")

        # Use Python's json module to load the mapping
        with open(mapping_path, 'r') as file:
            mapping = json.load(file)

        nones = []
        for item in mapping.keys():
            if mapping[item] is None:
                nones.append(item)

        self.preprocess.set_invalid_data(nones)

        return mapping

    def __call__(self, data, directory, threads) -> List[Tuple[Any, Dict[Any, Any]]]:
        data_unique = self._get_unique_data(data)
        if self.online:
            if self.preprocess is not None:
                data_unique = self.preprocess.data_preparations(
                    data_unique)
            mapping = (self.attribute, None)
            data = self._clean_data(data, data_unique)
            print(f"Preprocessing {self.attribute} done.")
            return data, mapping

        if self._preprocessing_done(directory):

            mapping_data = self._load_mapping(directory)
            if self.in_memory:
                path = os.path.join(
                    directory, f"{self.preprocessing_type}_{self.attribute}_mapping_info.json")
                data_path = os.path.join(
                    directory, f"{self.preprocessing_type}_{self.attribute}_all.pkl")
                processed_data = self.preprocess.load_preprocessed_to_memory(
                    data_path)

            new_data = list(set(data_unique) - set(mapping_data.keys()))

            if len(new_data) > 0:
                _ = self.preprocess.update(
                    processed_data, mapping_data, new_data, directory, self.in_memory, self.threads, f"{self.preprocessing_type}_{self.attribute}")

            if not self.in_memory:
                mapping_data = self._load_mapping(directory)
            else:
                mapping_data = self.preprocess.load_preprocessed_to_memory(
                    data_path)
                _ = self._load_mapping(directory)
        else:
            info_dict = self.preprocess.process_and_get_info(
                data_unique, directory, self.in_memory, threads, f"{self.preprocessing_type}_{self.attribute}")
            mapping_data = info_dict['mapping_info'] if not self.in_memory else info_dict['processed_data']

        mapping = (self.attribute, mapping_data)
        data = self._clean_data(data, data_unique)
        print(f"Preprocessing {self.attribute} done.")
        return data, mapping

    def __repr__(self):
        return f"PreprocessingObject(attribute={self.attribute}, preprocessing_type={self.preprocessing_type}, preprocessing_settings={self.preprocessing_settings}, in_memory={self.in_memory}, online={self.online})"

    def __add__(self, other):
        if not isinstance(other, PreprocessingObjectBase):
            raise TypeError(
                f"Cannot add object of type {type(other)} to subclass of PreprocessingObjectBase")

        return PreprocessingList([self, other])


class PreprocessingObject(PreprocessingObjectBase):
    def __init__(self, attribute: str, from_dtype: str, to_dtype: str, preprocessing_settings: Dict[str, Any], in_memory: bool = True, online: bool = False) -> None:
        super().__init__(attribute, from_dtype, to_dtype,
                         preprocessing_settings, in_memory, online)


T = TypeVar('T', bound='PreprocessingObjectBase')


class PreprocessingList:
    def __init__(self, p_list: List[T]) -> None:

        self.p_list = p_list
        self.attribute = [i.attribute for i in p_list]
        self.preprocessing_type = [i.preprocessing_type for i in p_list]
        self.preprocessing_settings = [
            i.preprocessing_settings for i in p_list]
        self.in_memory = [i.in_memory for i in p_list]
        self.online = [i.online for i in p_list]
        self.preprocess = [i.preprocess for i in p_list]
        self._check_same_type()

    def _check_same_type(self) -> None:
        for idx1, (i, j) in enumerate(zip(self.attribute, self.preprocessing_type)):
            for idx2, (k, l) in enumerate(zip(self.attribute, self.preprocessing_type)):
                if idx1 != idx2 and i == k and j == l:
                    if self.preprocessing_settings[idx1] != self.preprocessing_settings[idx2]:
                        new_name = self.preprocessing_type[idx2] + \
                            f"_setting_{idx2}"
                        self.preprocessing_type[idx2] = new_name

    def __call__(self, data: Any, directory: str, threads: int) -> Any:
        mappings = []
        for i in self.p_list:
            data, mapping = i(data, directory, threads)
            mappings.append(mapping)
        return data, mappings

    def __add__(self, other):
        if not isinstance(other, PreprocessingList) and not isinstance(other, PreprocessingObject):
            raise TypeError(
                f"Cannot add object of type {type(other)} to PreprocessingList")

        if isinstance(other, PreprocessingObject):
            other = PreprocessingList([other])

        new_object = PreprocessingList(self.p_list + other.p_list)
        return new_object

    def __iadd__(self, other):
        if not isinstance(other, PreprocessingObject):
            raise TypeError(
                f"Cannot add object of type {type(other)} to PreprocessingList")

        self.attribute += other.attribute
        self.preprocessing_type += other.preprocessing_type
        self.preprocessing_settings += other.preprocessing_settings
        self.in_memory += other.in_memory
        self.online += other.online
        self.preprocess += other.preprocess

        self._check_same_type()
        return self

    def __iter__(self):
        return iter(zip(self.attribute, self.preprocess, self.online, self.in_memory))

    def __len__(self):
        return len(self.attribute)

    def __getitem__(self, index):
        return self.attribute[index], self.preprocessing_type[index], self.preprocessing_settings[index], self.in_memory[index], self.online[index], self.preprocess[index]

    def __setitem__(self, index, value):
        self.attribute[index], self.preprocessing_type[index], self.preprocessing_settings[
            index], self.preprocess[index], self.in_memory[index], self.online[index] = value
        self._check_same_type()

    def _represent_preprocess(self, index):
        return f"Attribute={self.attribute[index]}, preprocessing_type={self.preprocessing_type[index]}, in_memory={self.in_memory[index]}, online={self.online[index]})\n"

    def __repr__(self) -> str:
        return f"PreprocessingList(\n{''.join([self._represent_preprocess(i) for i in range(len(self))])})"

    def get_attributes(self):
        return self.attribute


def has_data_preparation(sub_class):
    base_method = getattr(BasePreprocessor, "data_preparations", None)
    sub_method = getattr(sub_class, "data_preparations", None)

    if sub_method:
        return base_method.__qualname__ != sub_method.__qualname__
    return False


T = TypeVar('T', bound='BasePreprocessor')


def sequential_executor(*functions):
    """Executes a sequence of functions in order, where each function takes the result of the previous one as input.

    Args:
        *functions: A variable number of functions to execute sequentially.

    Returns:
        A new function that, when called, executes each of the provided functions in order, 
        passing the result of each function to the next.
    """
    def executor(initial_value):
        result = initial_value
        for function in functions:
            result = function(result)
        return result

    return executor


class SequentialPreprocessor(PreprocessingObject):
    """
    Class for sequential preprocessing, inheriting from PreprocessingObject.

    This class is intended to represent a sequence of preprocessing steps applied in order.
    Each step in the sequence should be an instance of a class derived from PreprocessingObjectBase.

    The actual implementation is to be completed (NotImplemented).
    """
    NotImplemented
