# todo: edit example usage

"""
This module defines abstract and custom dataset classes for use in drug-protein interaction projects.

The `AbstractDataset` class is an abstract base class (ABC) that outlines the structure for dataset handling,
including methods for downloading, loading, and processing datasets.

The `CustomDataset` class extends `AbstractDataset` and implements these methods. It is tailored for merging datasets
based on common columns and supports various preprocessing configurations. It can optionally associate with a model
and use its configuration for preprocessing.

Example:
    >>> custom_dataset = CustomDataset(file_paths=["./data/drug.csv", "./data/protein.csv"],
    ...                                drug_preprocess_type=[("scale", {})],
    ...                                protein_preprocess_type=[("encode", {"method": "one_hot"})],
    ...                                drug_attributes=["molecular_weight", "logP"],
    ...                                protein_attributes=["sequence"],
    ...                                common_columns=[{"DrugID": "DID"}, {"ProteinID": "PID"}],
    ...                                separators=[",", ","],
    ...                                save_directory="./processed_data",
    ...                                urls=["http://example.com/drug.csv", "http://example.com/protein.csv"],
    ...                                threads=4)
    >>> custom_dataset.download()
    >>> df = custom_dataset.load()
    >>> processed_dataset = custom_dataset()
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

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
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
    # todo: edit attributes
    """
    A dataset class that handles downloading, loading, and merging of datasets based on a common column.
    It supports preprocessing configuration and associates with a model if specified.

    Attributes:
        file_paths (List[str]): A list of file paths for the dataset files.
        drug_preprocess_type (Union[Tuple[str, Dict], List[Tuple[str, Dict]], None]): Preprocessing info for drugs.
        protein_preprocess_type (Union[Tuple[str, Dict], List[Tuple[str, Dict]], None]): Preprocessing info for proteins.
        save_directory (str, optional): Directory to save the processed datasets.
        urls (List[str], optional): List of URLs to download the datasets from.
        common_columns (List[Dict[str, str]], optional): Mapping for common columns across datasets.
        separators (List[str]): List of separators used in the dataset files.
        associated_model (str, optional): Model associated with the dataset for loading configurations.
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
        """Loads the model configuration from a JSON file."""

        if not ModelFactory.is_model_registered(associated_model):
            raise ValueError("Error: Model not found")

        config_path = os.path.join('configs', f'{associated_model}.json')
        with open(config_path, 'r') as config_file:
            model_configs = json.load(config_file)
            dataset_config = model_configs.get('dataset', {})

        self.drug_preprocess_type = drug_preprocess_type or dataset_config.get(
            'drug_preprocess_type')
        self.protein_preprocess_type = protein_preprocess_type or dataset_config.get(
            'protein_preprocess_type')

    def _validate_lengths(self):
        """Validates the lengths of the provided lists."""
        if self.urls and len(self.file_paths) != len(self.urls):
            raise ValueError("File paths and URLs must have the same length.")

        if len(self.file_paths) > 1 and self.common_columns and \
           len(self.file_paths) - 1 != len(self.common_columns):
            raise ValueError(
                "File paths - 1 and common columns must have the same length.")

    def download(self, *args, **kwargs) -> None:
        """Downloads dataset files from provided URLs to the specified file paths."""
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

    def load(self) -> Any:
        """Loads, merges, and returns a combined dataframe from the dataset files."""
        dfs = []
        for file_path, separator in zip(self.file_paths, self.separators):
            df = pd.read_csv(file_path, sep=separator)
            dfs.append(df)

        combined_df = dfs[0]
        for i, df in enumerate(dfs[1:], start=1):
            rename_mapping = self.common_columns[i - 1] if self.common_columns else {
            }
            combined_df = combined_df.merge(df.rename(columns=rename_mapping),
                                            on=list(rename_mapping.values()),
                                            how='inner')

        return combined_df

    def process(self) -> Any:
        """Processes the loaded dataset as required."""
        pass

    def __call__(self) -> Dataset:
        """Creates and returns a processed dataset ready for model consumption."""
        df = self.load()
        return DrugProteinDataset(df,
                                  self.drug_preprocess_type, self.drug_attributes, self.online_preprocessing_drug, self.in_memory_preprocessing_drug,
                                  self.protein_preprocess_type, self.protein_attributes, self.online_preprocessing_protein, self.in_memory_preprocessing_protein,
                                  self.label_attributes, self.label_preprocess_type, self.online_preprocessing_label, self.in_memory_preprocessing_label,
                                  self.save_directory,
                                  self.threads)
