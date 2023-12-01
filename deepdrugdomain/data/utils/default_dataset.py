
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..preprocessing.utils import PreprocessingList
from typing import List, Any, Optional, Dict, Tuple
from .dataset_utils import get_processed_data


class DDDDataset(Dataset):
    """
    A PyTorch Dataset for handling drug-protein interactions, incorporating preprocessed data.

    This dataset class is designed to work with a pandas DataFrame, applying preprocessing steps
    defined in a PreprocessingList object. It supports optional saving of processed data and
    multithreading for preprocessing.

    Attributes:
    ----------
    data: pd.DataFrame
        The dataset after applying the preprocessing steps.
    mapping: List[Tuple[Any, Dict[Any, Any]]]
        A list of tuples containing attribute mappings as a result of preprocessing.
    preprocesses: PreprocessingList
        The PreprocessingList object containing preprocessing steps to be applied.
    save_directory: Optional[str]
        The directory where preprocessed data can be saved. If None, data is not saved.
    threads: int
        The number of threads to use for preprocessing.

    Parameters:
    ----------
    data: pd.DataFrame
        The original pandas DataFrame containing the dataset.
    preprocesses: PreprocessingList
        An instance of PreprocessingList defining the preprocessing pipeline.
    save_directory: Optional[str], default=None
        A directory path to save the preprocessed data. If None, data is not saved.
    threads: int, default=4
        The number of threads to use for parallel preprocessing.

    Methods:
    -------
    __len__() -> int:
        Returns the length of the dataset.
    __getitem__(index: int) -> Any:
        Retrieves the preprocessed data at the specified index.
    
    Example:
    -------
    Assume 'df' is a pandas DataFrame containing your dataset, and 'preprocesses' is an 
    instance of PreprocessingList prepared with relevant preprocessing steps.

    >>> dataset = DrugProteinDataset(df, preprocesses)
    >>> print(f"Dataset length: {len(dataset)}")
    >>> first_item = dataset[0]
    >>> print(first_item)
    """
    def __init__(self,
                 data: pd.DataFrame,
                 preprocesses: PreprocessingList,
                 save_directory: Optional[str] = None,
                 threads: int = 4) -> None:

        self.save_directory = save_directory
        self.threads = threads
        self.data, self.mapping = preprocesses(data, save_directory, threads)
        self.preprocesses = preprocesses

    def __len__(self) -> int:
        return len(self.data.index)

    def __getitem__(self, index: int) -> Any:
        attrs = zip(self.preprocesses, self.mapping)
        data = []
        for (_, pre_process, online, in_mem), mapping in attrs:
            if mapping[0] is None:
                continue
            else:
                data.append(get_processed_data(online, mapping, pre_process,
                                               in_mem, self.data.iloc[index][mapping[0]]))

        return data
