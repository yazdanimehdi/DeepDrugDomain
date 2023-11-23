from typing import Dict, Tuple, Type, TypeVar, List
from ..utils.base_dataset import AbstractDataset
from deepdrugdomain.utils import BaseFactory
from ..utils.data_struct import PreprocessingObject
T = TypeVar('T', bound=AbstractDataset)


class DatasetFactory(BaseFactory):
    """
    A factory class for creating dataset instances.

    This class maintains a registry of dataset types, enabling the dynamic creation of datasets based on a string key. 
    The datasets created can be used for various machine learning tasks and are especially tailored for drug discovery 
    and protein interaction applications.

    Attributes:
        _registry (Dict[str, Type[AbstractDataset]]): A private dictionary that maps string keys to dataset classes.

    Example:
        >>> @DatasetFactory.register('my_dataset')
        ... class MyDataset(AbstractDataset):
        ...     # Implementation of MyDataset
        ...
        >>> dataset = DatasetFactory.create(
        ...     'my_dataset',
        ...     file_paths='path/to/data',
        ...     preprocesses=PreprocessingObject()
        ... )
        This example registers a new dataset class 'MyDataset' under the key 'my_dataset' and then instantiates it.

    Methods:
        register(key: str): A class method decorator used for adding dataset classes to the factory's registry.
        create(key: str, ...): Creates an instance of a dataset class corresponding to the given key with the option 
                               to pass additional arguments and keyword arguments specific to the dataset class.
    """
    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, key: str):
        """
        Registers a dataset class to the factory under a specific key.

        Args:
            key (str): The key associated with the dataset class to be registered.

        Returns:
            A decorator function that registers the given dataset subclass to the factory's registry.

        Raises:
            TypeError: If the given subclass is not a subclass of AbstractDataset.
        """

        def decorator(subclass: Type[T]) -> Type[T]:
            if not issubclass(subclass, AbstractDataset):
                raise TypeError(
                    f"Class {subclass.__name__} is not a subclass of AbstractDataset")
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls,
               key: str,
               file_paths: str,
               preprocesses: PreprocessingObject,
               **kwargs) -> Type[T]:
        """
        Creates an instance of the dataset associated with the given key.

        Args:
            key (str): The key for the dataset class to instantiate.
            file_paths (str): The file paths for dataset files.
            drug_preprocess_type (Tuple[str, Dict] | List[Tuple[str, Dict] | None] | None): Preprocessing configurations for drug data.
            protein_preprocess_type (Tuple[str, Dict] | List[Tuple[str, Dict] | None] | None): Preprocessing configurations for protein data.
            protein_attributes (List[str] | str): Attributes to be considered for proteins.
            in_memory_preprocessing_protein (List[bool] | bool): Flags indicating whether to preprocess protein data in memory.
            **kwargs: Keyword arguments for the dataset constructor.

        Returns:
            An instance of the dataset class associated with the given key.

        Raises:
            ValueError: If the key is not registered in the factory's registry.
        """

        if key not in cls._registry:
            raise ValueError(f"Key '{key}' not registered.")

        dataset_instance = cls._registry[key](
            file_paths, preprocesses, **kwargs)
        return dataset_instance
