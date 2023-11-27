from typing import Dict, Any, Union, List
from ..preprocessing import PreprocessorFactory, BasePreprocessor
from .dataset_utils import ensure_list


class PreprocessingObject:
    """
    A class to represent preprocessing configurations for data attributes.

    This class encapsulates information about the preprocessing to be applied to data attributes, 
    including the type of preprocessing, its specific settings, and the modes (in-memory and online) 
    in which it should be executed.

    Attributes:
    -----------
    attribute (List[str]):
        Names or identifiers of the attributes.

    preprocessing_type (List[Union[str, BasePreprocessor]]):
        Types of preprocessing to be applied. Each element can be a string (name of the preprocessor)
        or an instance of a BasePreprocessor.

    preprocessing_settings (List[Dict[str, Any]]):
        Settings for each preprocessing type.

    in_memory (List[bool]):
        Flags indicating whether each preprocessing should be done in memory.

    online (List[bool]):
        Flags indicating whether each preprocessing should be done online.

    Methods:
    --------
    __add__(other):
        Adds the preprocessing configurations from another PreprocessingObject.

    __iadd__(other):
        In-place addition of preprocessing configurations from another PreprocessingObject.

    __iter__():
        Returns an iterator over the combined preprocessing configurations.

    __len__():
        Returns the number of preprocessing configurations.

    __getitem__(index):
        Accesses the preprocessing configuration at the specified index.

    __setitem__(index, value):
        Sets the preprocessing configuration at the specified index.

    Examples:
    ---------
    >>> preprocess_obj = PreprocessingObject('attr1', 'type1', {'setting1': 10}, True, False)
    >>> print(preprocess_obj)
    PreprocessingObject(attribute=['attr1'], preprocessing_type=['type1'], preprocessing_settings=[{'setting1': 10}], in_memory=[True], online=[False])

    >>> preprocess_obj2 = PreprocessingObject(['attr2', 'attr3'], ['type2', 'type3'], [{'setting2': 20}, {'setting3': 30}], [False, True], [True, False])
    >>> combined_obj = preprocess_obj + preprocess_obj2
    >>> print(combined_obj)
    PreprocessingObject(attribute=['attr1', 'attr2', 'attr3'], preprocessing_type=['type1', 'type2', 'type3'], preprocessing_settings=[{'setting1': 10}, {'setting2': 20}, {'setting3': 30}], in_memory=[True, False, True], online=[False, True, False])
    """

    def __init__(self, attribute: Union[str, List[str]], preprocessing_type: Union[str, List[str]], preprocessing_settings: Union[Dict[str, Any], List[Dict[str, Any]]], in_memory: Union[bool, List[bool]] = True, online: Union[bool, List[bool]] = False) -> None:
        """
        Initializes a new instance of the PreprocessingObject class.

        This class holds preprocessing configurations for data attributes, including
        the type of preprocessing, settings, and modes (in-memory and online).

        Args:
        ----
        attribute (Union[str, List[str]]):
            The name or identifier of the attribute(s). Can be a single string or a list of strings.

        preprocessing_type (Union[str, List[str], List[BasePreprocessor]]):
            The type of preprocessing to be applied. Can be a string (name of the preprocessor),
            a list of strings, or a list of Preprocessor instances.

        preprocessing_settings (Union[Dict[str, Any], List[Dict[str, Any]]]):
            The settings for the preprocessing. Can be a dictionary for a single preprocessing type
            or a list of dictionaries for multiple types.

        in_memory (Union[bool, List[bool]], optional):
            Indicates whether the preprocessing should be done in memory.
            Defaults to True. Can be a single boolean or a list of booleans.

        online (Union[bool, List[bool]], optional):
            Indicates whether the preprocessing should be done online.
            Defaults to False. Can be a single boolean or a list of booleans.
        """

        if in_memory is None:
            in_memory = True
        if online is None:
            online = True

        self.attribute = ensure_list(attribute)
        self.preprocessing_type = ensure_list(preprocessing_type)
        self.preprocessing_settings = ensure_list(preprocessing_settings)
        self.in_memory = ensure_list(in_memory)
        self.online = ensure_list(online)
        self.preprocess = []

        self._check_same_type()
        self.validate_and_initialize()

    def __repr__(self):
        return f"PreprocessingObject(attribute={self.attribute}, preprocessing_type={self.preprocessing_type}, preprocessing_settings={self.preprocessing_settings}, in_memory={self.in_memory}, online={self.online})"

    def _check_same_type(self) -> None:
        for idx1, (i, j) in enumerate(zip(self.attribute, self.preprocessing_type)):
            for idx2, (k, l) in enumerate(zip(self.attribute, self.preprocessing_type)):
                if idx1 != idx2 and i == k and j == l:
                    new_name = self.preprocessing_type[idx2] + \
                        f"_setting_{idx2}"
                    self.preprocessing_type[idx2] = new_name

    def validate_and_initialize(self):
        """
        Validates and initializes the preprocessing configuration.

        Ensures that all attributes are of the same length and initializes
        the preprocessing types if given as string identifiers.
        """
        max_length = len(self.attribute)
        self.preprocessing_type += [None] * \
            (max_length - len(self.preprocessing_type))
        self.preprocessing_settings += [{}] * \
            (max_length - len(self.preprocessing_settings))
        self.in_memory += [True] * (max_length - len(self.in_memory))
        self.online += [False] * (max_length - len(self.online))

        assert len(self.attribute) == len(self.preprocessing_type) == len(self.preprocessing_settings) == len(
            self.in_memory) == len(self.online), "All attributes must have the same length"

        self.preprocess = []
        for i, pre in enumerate(self.preprocessing_type):
            if "setting" in pre:
                pre = pre.split("_setting")[0]
            self.preprocess.append(PreprocessorFactory.create(
                pre, **self.preprocessing_settings[i]))

    def __add__(self, other):
        if not isinstance(other, PreprocessingObject):
            raise TypeError(
                f"Cannot add object of type {type(other)} to PreprocessingObject")

        new_object = PreprocessingObject(
            self.attribute, self.preprocessing_type, self.preprocessing_settings, self.in_memory, self.online)
        new_object.attribute += other.attribute
        new_object.preprocessing_type += other.preprocessing_type
        new_object.preprocessing_settings += other.preprocessing_settings
        new_object.in_memory += other.in_memory
        new_object.online += other.online
        new_object.validate_and_initialize()
        self._check_same_type()
        return new_object

    def __iadd__(self, other):
        if not isinstance(other, PreprocessingObject):
            raise TypeError(
                f"Cannot add object of type {type(other)} to PreprocessingObject")

        self.attribute += other.attribute
        self.preprocessing_type += other.preprocessing_type
        self.preprocessing_settings += other.preprocessing_settings
        self.in_memory += other.in_memory
        self.online += other.online
        self._check_same_type()
        self.validate_and_initialize()
        return self

    def __iter__(self):
        return iter(zip(self.attribute, self.preprocessing_type, self.preprocess, self.in_memory, self.online))

    def __len__(self):
        return len(self.attribute)

    def __getitem__(self, index):
        return self.attribute[index], self.preprocessing_type[index], self.preprocessing_settings[index], self.in_memory[index], self.online[index]

    def __setitem__(self, index, value):
        self.attribute[index], self.preprocessing_type[index], self.preprocessing_settings[index], self.in_memory[index], self.online[index] = value

    def get_attributes(self):
        return self.attribute
