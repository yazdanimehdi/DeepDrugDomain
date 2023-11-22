import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from ..factory import DatasetFactory


@DatasetFactory.register('human')
class HumanDataset(CustomDataset):
    """
    Dataset class for human drug-target interaction data.

    This class extends CustomDataset to provide a structured way to load and preprocess human drug-target interaction 
    datasets. It enables the integration of drug and protein data, along with their corresponding labels, for tasks 
    such as interaction prediction or drug repurposing.

    Parameters:
        file_paths (str): Directory path where data files are stored or to be downloaded.
        drug_preprocess_type (Union[Tuple[str, Dict], List[Tuple[str, Dict]], None]): Preprocessing configuration(s) 
            for drug data.
        protein_preprocess_type (Union[Tuple[str, Dict], List[Tuple[str, Dict]], None]): Preprocessing configuration(s) 
            for protein data.
        protein_attributes (Union[List[str], str]): Attributes to be considered for proteins.
        in_memory_preprocessing_protein (Union[List[bool], bool]): Flags indicating whether to preprocess protein data 
            in memory.
        drug_attributes (Union[List[str], str], optional): Attributes to be considered for drugs, defaults to "SMILES".
        online_preprocessing_drug (Union[List[bool], bool], optional): Flags indicating whether to preprocess drug data 
            online (on-the-fly).
        in_memory_preprocessing_drug (Union[List[bool], bool], optional): Flags indicating whether to preprocess drug 
            data in memory.
        online_preprocessing_protein (Union[List[bool], bool], optional): Flags indicating whether to preprocess protein 
            data online (on-the-fly).
        label_attributes (Union[List[str], str], optional): Attributes for the labels.
        label_preprocess_type (Union[Tuple[str, Dict], List[Tuple[str, Dict]], None], optional): Preprocessing 
            configuration(s) for label data.
        online_preprocessing_label (Union[List[bool], bool], optional): Flags indicating whether to preprocess label 
            data online.
        in_memory_preprocessing_label (Union[List[bool], bool], optional): Flags indicating whether to preprocess label 
            data in memory.
        save_directory (Optional[str]): The directory to save processed files, defaults to `file_paths` if None.
        urls (Optional[Union[List[str], str]]): URLs to download the dataset files if not present at `file_paths`.
        common_columns (Optional[Union[Dict[str, str], List[Dict[str, str]]]]): Mapping of common column names to the 
            expected format.
        separators (Union[List[str], str], optional): List of separators used in the data files.
        associated_model (Optional[str]): The name of the model associated with the dataset, if any.
        threads (int, optional): Number of threads to use for data processing.

    Example:
        >>> dataset = HumanDataset(
        ...     file_paths='/data/human/',
        ...     drug_preprocess_type=('canonical_smiles', {'remove_hydrogens': True}),
        ...     protein_preprocess_type=('Target_Seq', {'tokenization': 'char'}),
        ...     protein_attributes='sequence',
        ...     in_memory_preprocessing_protein=True,
        ... )
        >>> train_dataset, val_dataset, test_dataset = dataset(splits=[0.8, 0.1, 0.1], return_df=False)  # Preprocess and split the dataset into train, validation, and test sets and prepare data for training or analysis
        >>> human_dataframe = dataset(return_df=True)  # Get the raw dataset as a pandas DataFrame

    Note:
        The class automatically downloads the necessary files if they are not available in the given `file_paths` 
        during initialization.
    """

    def __init__(self, file_paths: str,
                 drug_preprocess_type: Tuple[str, Dict] | List[Tuple[str, Dict] | None] | None,
                 protein_preprocess_type: Tuple[str, Dict] | List[Tuple[str, Dict] | None] | None,
                 protein_attributes: List[str] | str,
                 in_memory_preprocessing_protein: List[bool] | bool,
                 drug_attributes: List[str] | str = "SMILES",
                 online_preprocessing_drug: List[bool] | bool = False,
                 in_memory_preprocessing_drug: List[bool] | bool = True,
                 online_preprocessing_protein: List[bool] | bool = False,
                 label_attributes: List[str] | str = 'Label',
                 label_preprocess_type: Tuple[str,
                                              Dict] | List[Tuple[str, Dict] | None] | None = None,
                 online_preprocessing_label: List[bool] | bool = True,
                 in_memory_preprocessing_label: List[bool] | bool = True,
                 save_directory: str | None = None,
                 urls: List[str] | str | None = ['https://github.com/yazdanimehdi/DeepDrugDomain/raw/main/data/human/human_data.txt',
                                                 'https://github.com/yazdanimehdi/DeepDrugDomain/raw/main/data/human/humanSeqPdb.txt'],
                 common_columns: Dict[str,
                                      str] | List[Dict[str, str]] | None = {'sequence': 'Target_Seq'},
                 separators: List[str] | str = [' ', ','],
                 associated_model: str | None = None,
                 threads: int = 4) -> None:

        human_data_path = os.path.join(file_paths, 'human_data.txt')
        humanSeqPdb_path = os.path.join(file_paths, 'humanSeqPdb.txt')
        file_paths_new = [human_data_path, humanSeqPdb_path]

        save_directory = file_paths if save_directory is None else save_directory

        if not os.path.exists(human_data_path) or not os.path.exists(humanSeqPdb_path):
            self.download()

        super().__init__(file_paths_new, drug_preprocess_type, drug_attributes, online_preprocessing_drug, in_memory_preprocessing_drug, protein_preprocess_type, protein_attributes, online_preprocessing_protein,
                         in_memory_preprocessing_protein, label_attributes, label_preprocess_type, online_preprocessing_label, in_memory_preprocessing_label, save_directory, urls, common_columns, separators, associated_model, threads)
