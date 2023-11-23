import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from deepdrugdomain.data.utils.data_struct import PreprocessingObject
from ..factory import DatasetFactory


@DatasetFactory.register('celegans')
class CelegansDataset(CustomDataset):
    """
        Dataset class for C. elegans (a model organism) drug-target interaction data.

        This class extends CustomDataset to provide a structured way to load and preprocess C. elegans drug-target interaction 
        datasets. It enables the integration of drug and protein data, along with their corresponding labels, for tasks such as 
        interaction prediction or drug repurposing within the scope of C. elegans research.

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
            >>> dataset = CelegansDataset(
            ...     file_paths='/data/celegans/',
            ...     drug_preprocess_type=('canonical_smiles', {'remove_hydrogens': True}),
            ...     protein_preprocess_type=('sequence', {'tokenization': 'char'}),
            ...     protein_attributes='sequence',
            ...     in_memory_preprocessing_protein=True,
            ... )
            >>> train_dataset, val_dataset, test_dataset = dataset.split(splits=[0.8, 0.1, 0.1], return_df=False) 
            >>> # Preprocess and split the dataset into train, validation, and test sets and prepare data for training or analysis
            >>> celegans_dataframe = dataset(return_df=True)  # Get the raw dataset as a pandas DataFrame

        Note:
            The class automatically downloads the necessary files if they are not available in the given `file_paths` during 
            initialization, using the provided `urls` for data source.
    """

    def __init__(self, file_paths: str,
                 preprocesses: PreprocessingObject,
                 save_directory: str | None = None,
                 urls: List[str] | str | None = ['https://github.com/yazdanimehdi/DeepDrugDomain/raw/main/data/celegans/true.txt',
                                                 'https://github.com/yazdanimehdi/DeepDrugDomain/raw/main/data/celegans/celegansSeqPdb.txt'],
                 common_columns: Dict[str,
                                      str] | List[Dict[str, str]] | None = {'sequence': 'Target_Seq'},
                 separators: List[str] | str = [' ', ','],
                 associated_model: str | None = None,
                 threads: int = 4) -> None:

        self.file_paths = file_paths
        celegans_data_path = os.path.join(self.file_paths, 'true.txt')
        celegansSeqPdb_path = os.path.join(
            self.file_paths, 'celegansSeqPdb.txt')
        file_paths = [celegans_data_path, celegansSeqPdb_path]
        save_directory = self.file_paths if save_directory is None else save_directory

        super().__init__(file_paths, preprocesses, save_directory, urls,
                         common_columns, separators, associated_model, None, threads)

        if not os.path.exists(celegans_data_path) or not os.path.exists(celegansSeqPdb_path):
            self.download()
