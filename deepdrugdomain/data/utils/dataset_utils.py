import os
import sys
from typing import Any, List, Union, Dict, Tuple


def estimate_sample_size(sample: Any) -> int:
    """
    Estimate memory usage of a single sample in bytes.

    Parameters:
    - sample (Any): A single data sample.

    Returns:
    - int: Estimated size of the sample in bytes.
    """
    # This is a basic estimation. Depending on the structure of 'sample', a more complex calculation might be needed.
    return sys.getsizeof(sample)


def ensure_list(value: Union[List[Any], Any]) -> List[Any]:
    """
    Ensure the provided value is returned as a list.

    If the input value is not already a list, it is wrapped into a list.
    If it's already a list, it's returned as-is.

    Parameters:
    - value (Union[List[Any], Any]): The input value which can be of any type or a list of any type.

    Returns:
    - List[Any]: A list containing the input value(s).

    Example:
    >>> ensure_list("hello")
    ['hello']

    >>> ensure_list(["hello", "world"])
    ['hello', 'world']

    >>> ensure_list(5)
    [5]
    """

    # Check if the input value is not already a list
    if not isinstance(value, list):
        return [value]  # Wrap the value into a list

    # If the input is already a list, return it as-is
    return value


def assert_unique_combinations(list1: list, list2: list, list3: list) -> None:
    """
    Assert that the three lists are of the same length and that no ordered combination
    between them (one element from each list) is repeated.

    Parameters:
    -----------
    list1 : list
        The first list for comparison.

    list2 : list
        The second list for comparison.

    list3 : list
        The third list for comparison.

    Raises:
    -------
    AssertionError
        If the lists don't have the same length or if there's a repeated combination for list1 and list2.

    Examples:
    ---------
    >>> l1 = [3, 2, 3]
    >>> l2 = ['a', 'b', 'a']
    >>> l3 = [1.1, 2.2, 3.3]
    >>> assert_unique_combinations(l1, l2, l3)
    AssertionError: Repeated combination: 3, a

    >>> l1 = [1, 2]
    >>> l2 = ['a', 'b', 'c']
    >>> l3 = [1.1, 2.2, 3.3]
    >>> assert_unique_combinations(l1, l2, l3)
    AssertionError: All lists must have the same length.
    """

    # Assert that all lists have the same length
    assert len(list1) == len(list2) == len(
        list3), "All lists must have the same length."

    # Use a set to track the combinations
    # seen_combinations = set()
    #
    # for a, b in zip(list1, list2):
    #     # If this combination has been seen before, raise an error
    #     assert (a, b) not in seen_combinations, f"Repeated combination: {a}, {b}"
    #
    #     # Add the current combination to the set
    #     seen_combinations.add((a, b))


def get_processed_data(online: bool, mapping: Tuple[str, Any], pre_process, in_mem, row_data):
    if pre_process is None:
        return row_data

    if online:
        return pre_process.preprocess(row_data)

    elif in_mem:
        return mapping[1][row_data]
    else:
        path = mapping[1][row_data]
        if os.path.exists(path):  # Check if shard exists before loading
            return pre_process.load_data(path)
        else:
            raise FileNotFoundError(f"Shard file {path} not found.")
