import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from deepdrugdomain.data.utils.data_struct import PreprocessingObject
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
        preprocesses (PreprocessingObject): Preprocessing configuration(s) for drug, protein, and label data.
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
                 preprocesses: PreprocessingObject,
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
        super().__init__(file_paths_new, preprocesses, save_directory,
                         urls, common_columns, separators, associated_model, None, threads)

        if not os.path.exists(human_data_path) or not os.path.exists(humanSeqPdb_path):
            self.download()
