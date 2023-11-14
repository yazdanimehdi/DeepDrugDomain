"""
This module defines abstract and custom dataset classes for use in drug-protein interaction projects.

The `AbstractDataset` class is an abstract base class (ABC) that outlines the structure for dataset handling,
including methods for downloading, loading, processing, and calling datasets as objects.

The `CustomDataset` class extends `AbstractDataset` and implements these methods. It is tailored for merging datasets
based on common columns and supports various preprocessing configurations. It can optionally associate with a model
and use its configuration for preprocessing.

The usage of the `CustomDataset` class is demonstrated below, showing how it can be used to create training, validation,
and test datasets from specified file paths and processing configurations.

Example:
    >>> dataset = CustomDataset(
    ...     file_paths=["data/drugbank/DrugBank.txt", "data/drugbank/drugbankSeqPdb.txt"],
    ...     common_columns={"sequence": "TargetSequence"},
    ...     separators=[" ", ","],
    ...     drug_preprocess_type=("dgl_graph_from_smile", {"fragment": False, "max_block": 6, "max_sr": 8, "min_frag_atom": 1}),
    ...     drug_attributes="SMILE",
    ...     online_preprocessing_drug=False,
    ...     in_memory_preprocessing_drug=True,
    ...     protein_preprocess_type=("dgl_graph_from_protein_pocket", {"pdb_path": "data/pdb/", "protein_size_limit": 10000}),
    ...     protein_attributes="pdb_id",
    ...     online_preprocessing_protein=False,
    ...     in_memory_preprocessing_protein=False,
    ...     label_attributes="Label",
    ...     save_directory="data/drugbank/",
    ...     threads=8
    ... )
    >>> dataset_train, dataset_val, dataset_test = dataset.random_split([0.8, 0.1, 0.1])
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional, Union, Tuple
import pandas as pd
import os
import json
from ..utils import ensure_list
from tqdm import tqdm
import requests
from deepdrugdomain.models.factory import ModelFactory
from .default_dataset import DrugProteinDataset
from torch.utils.data import Dataset
import torch


class AbstractDataset(ABC):

    @classmethod
    @abstractmethod
    def download(cls, *args, **kwargs) -> None:
        """
        Downloads the dataset from a remote location. This is an optional method which can be implemented by subclasses.
        """
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """
        Loads the dataset from the local filesystem.
        """
        pass

    @classmethod
    @abstractmethod
    def process_file(self, *args, **kwargs) -> Any:
        """
        Processes the dataset.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Calls the dataset object.
        """
        pass


class CustomDataset(AbstractDataset):
    """
    A dataset class that handles the complexities of setting up a drug-protein interaction dataset. This class manages
    downloading, loading, and merging datasets from multiple sources, ensuring that they align on a common set of columns.
    It offers support for preprocessing configurations for both drug and protein data and can be linked to an associated
    model's configuration.

    Attributes:
        file_paths (List[str]): List of file paths for the dataset files.
        drug_preprocess_type (Optional[Union[Tuple[str, Dict], List[Tuple[str, Dict]]]]): Preprocessing information for drugs.
        drug_attributes (Union[List[str], str]): Attributes to extract for drugs from the dataset.
        online_preprocessing_drug (Union[List[bool], bool]): Flags to determine if drugs need online preprocessing.
        in_memory_preprocessing_drug (Union[List[bool], bool]): Flags to determine if drugs need in-memory preprocessing.
        protein_preprocess_type (Optional[Union[Tuple[str, Dict], List[Tuple[str, Dict]]]]): Preprocessing information for proteins.
        protein_attributes (Union[List[str], str]): Attributes to extract for proteins from the dataset.
        online_preprocessing_protein (Union[List[bool], bool]): Flags to determine if proteins need online preprocessing.
        in_memory_preprocessing_protein (Union[List[bool], bool]): Flags to determine if proteins need in-memory preprocessing.
        label_attributes (Union[List[str], str]): Attributes used for the labels in the dataset.
        label_preprocess_type (Optional[Union[Tuple[str, Dict], List[Tuple[str, Dict]]]]): Preprocessing information for labels.
        online_preprocessing_label (Union[List[bool], bool]): Flags to determine if labels need online preprocessing.
        in_memory_preprocessing_label (Union[List[bool], bool]): Flags to determine if labels need in-memory preprocessing.
        save_directory (Optional[str]): Directory to save the processed datasets.
        urls (Optional[Union[List[str], str]]): URLs from where to download the datasets.
        common_columns (Optional[Union[Dict[str, str], List[Dict[str, str]]]]): Common columns mapping for merging datasets.
        separators (Union[List[str], str]): Column separators for reading the datasets.
        associated_model (Optional[str]): The model associated with the dataset for loading configurations.
        threads (int): Number of threads to use for downloading and processing.
    """

    def __init__(self,
                 file_paths: Union[List[str], str],
                 drug_preprocess_type: Optional[Union[Union[Tuple[str, Dict], None], List[Union[Tuple[str, Dict], None]]]],
                 drug_attributes: Union[List[str], str],
                 online_preprocessing_drug: Union[List[bool], bool],
                 in_memory_preprocessing_drug: Union[List[bool], bool],

                 protein_preprocess_type: Optional[Union[Union[Tuple[str, Dict], None], List[Union[Tuple[str, Dict], None]]]],
                 protein_attributes: Union[List[str], str],
                 online_preprocessing_protein: Union[List[bool], bool],
                 in_memory_preprocessing_protein: Union[List[bool], bool],

                 label_attributes: Union[List[str], str],
                 label_preprocess_type: Union[Union[Tuple[str, Dict],
                                                    None], List[Union[Tuple[str, Dict], None]]] = None,
                 online_preprocessing_label: Union[List[bool], bool] = True,
                 in_memory_preprocessing_label: Union[List[bool], bool] = True,

                 save_directory: Optional[str] = None,
                 urls: Optional[Union[List[str], str]] = None,
                 common_columns: Optional[Union[Dict[str,
                                                     str], List[Dict[str, str]]]] = None,
                 separators: Union[List[str], str] = ',',
                 associated_model: Optional[str] = None,
                 threads: int = 4) -> None:
        super().__init__()
        """
            Initializes the CustomDataset object with the given parameters. Validates the lengths of provided lists
            and loads the model configuration if an associated model is provided.
        """
        self.file_paths = ensure_list(file_paths)
        self.urls = ensure_list(urls) if urls else None
        self.common_columns = ensure_list(common_columns)
        self.threads = threads
        self.save_directory = save_directory
        self.separators = ensure_list(separators)
        self.drug_attributes = ensure_list(drug_attributes)
        self.protein_attributes = ensure_list(protein_attributes)
        self.online_preprocessing_drug = ensure_list(
            online_preprocessing_drug)
        self.online_preprocessing_protein = ensure_list(
            online_preprocessing_protein)
        self.in_memory_preprocessing_drug = ensure_list(
            in_memory_preprocessing_drug)
        self.in_memory_preprocessing_protein = ensure_list(
            in_memory_preprocessing_protein)
        self.drug_preprocess_type = ensure_list(drug_preprocess_type)
        self.protein_preprocess_type = ensure_list(protein_preprocess_type)
        self.label_attributes = ensure_list(label_attributes)
        self.online_preprocessing_label = ensure_list(
            online_preprocessing_label)
        self.in_memory_preprocessing_label = ensure_list(
            in_memory_preprocessing_label)
        self.label_preprocess_type = ensure_list(label_preprocess_type)

        if associated_model:
            self._load_model_config(associated_model)

        self._validate_lengths()

    def _load_model_config(self, associated_model, drug_preprocess_type, protein_preprocess_type):
        """
            Load model configuration for preprocessing from a specified JSON file. If specific preprocess types for drugs or proteins 
            are not provided, the method attempts to load these configurations from the model's config file.

            Parameters:
                associated_model (str): The name of the model whose configuration is to be loaded.
                drug_preprocess_type (Optional[Union[Tuple[str, Dict], List[Tuple[str, Dict]]]]): Provided drug preprocessing information.
                protein_preprocess_type (Optional[Union[Tuple[str, Dict], List[Tuple[str, Dict]]]]): Provided protein preprocessing information.

            Raises:
                ValueError: If the specified model is not registered in the ModelFactory.
        """
        if not ModelFactory.is_model_registered(associated_model):
            raise ValueError("Error: Model not found")

        config_path = os.path.join('configs', f'{associated_model}.json')
        assert os.path.exists(config_path), "Model config file not found"

        try:
            with open(config_path, 'r') as config_file:
                model_configs = json.load(config_file)
                dataset_config = model_configs.get('dataset', {})

            dataset_config = dataset_config[self.__class__.__name__]
        except KeyError:
            raise ValueError(
                f"Error: Model config file does not contain a dataset configuration for {self.__class__.__name__}")

        self.drug_preprocess_type = drug_preprocess_type or dataset_config.get(
            'drug_preprocess_type')
        self.protein_preprocess_type = protein_preprocess_type or dataset_config.get(
            'protein_preprocess_type')

    def _validate_lengths(self):
        """
            Validates that the provided lists of file paths, URLs, and common columns have compatible lengths. This ensures that each file path 
            has a corresponding URL and common column information for merging datasets.

            Raises:
                ValueError: If the lengths of file paths, URLs, and common columns do not match.
        """
        if self.urls and len(self.file_paths) != len(self.urls):
            raise ValueError("File paths and URLs must have the same length.")

        if len(self.file_paths) > 1 and self.common_columns and \
           len(self.file_paths) - 1 != len(self.common_columns):
            raise ValueError(
                "File paths - 1 and common columns must have the same length.")

    def download(self, *args, **kwargs) -> None:
        """
            Downloads dataset files from their respective URLs to the specified file paths. If a file already exists at a given path, 
            the download is skipped for that file. If the download fails or the file is only partially downloaded, it is removed.

            Raises:
                requests.exceptions.RequestException: If an HTTP request exception occurs during file download.
        """
        if self.urls is None:
            return

        for url, file_path in zip(self.urls, self.file_paths):
            if os.path.exists(file_path):
                print(
                    f"File at {file_path} already exists, skipping download.")
                continue

            try:
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(
                    response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kilobyte

                with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {url}") as progress_bar:
                    with open(file_path, 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print("ERROR, something went wrong with the download")
                        os.remove(file_path)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred while downloading {url}: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path)

    def load(self) -> pd.DataFrame:
        """
            Loads multiple datasets from the provided file paths using their respective separators and merges them into a single DataFrame 
            based on the mappings defined in 'common_columns'. It uses 'process_file' to process each individual file.

            Returns:
                pd.DataFrame: The merged DataFrame containing all data from the provided files.
        """
        dfs = []
        for file_path, separator in zip(self.file_paths, self.separators):
            df = self.process_file(file_path, separator)
            dfs.append(df)

        combined_df = dfs[0]
        for i, df in enumerate(dfs[1:], start=1):
            rename_mapping = self.common_columns[i - 1] if self.common_columns else {
            }
            combined_df = combined_df.merge(df.rename(columns=rename_mapping),
                                            on=list(rename_mapping.values()),
                                            how='inner')

        return combined_df

    @classmethod
    def process_file(self, file_path, separator) -> pd.DataFrame:
        """
            Reads a CSV file into a DataFrame using the provided separator. This class method can be overridden by subclasses to 
            implement custom file reading logic.

            Parameters:
                file_path (str): The path to the CSV file to be processed.
                separator (str): The delimiter to use when parsing the CSV file.

            Returns:
                pd.DataFrame: The DataFrame obtained from the CSV file.
        """
        return pd.read_csv(file_path, sep=separator)

    def __call__(self, random_split: Optional[List[int]] = None, return_df: Optional[bool] = False) -> Union[Dataset, pd.DataFrame]:
        """
            When called, the method creates a DrugProteinDataset from the loaded data and optionally splits it into training, 
            validation, and test datasets based on the provided proportions in 'random_split'.

            Parameters:
                random_split (Optional[List[float]]): A list of proportions for splitting the dataset. If provided, the sum of proportions 
                                                    must equal 1.

            Returns:
                Union[Dataset, Tuple[Dataset, ...]]: The complete dataset or a tuple containing split datasets.
        """
        df = self.load()

        if return_df:
            return df

        dataset = DrugProteinDataset(df.head(100),
                                     self.drug_preprocess_type, self.drug_attributes, self.online_preprocessing_drug, self.in_memory_preprocessing_drug,
                                     self.protein_preprocess_type, self.protein_attributes, self.online_preprocessing_protein, self.in_memory_preprocessing_protein,
                                     self.label_attributes, self.label_preprocess_type, self.online_preprocessing_label, self.in_memory_preprocessing_label,
                                     self.save_directory,
                                     self.threads)

        return torch.utils.data.random_split(dataset, random_split) if random_split else dataset
