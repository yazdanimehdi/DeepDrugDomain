import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from deepdrugdomain.data.utils import DatasetFactory


@DatasetFactory.register('TDC')
class TDC(CustomDataset):
    def __init__(self, file_paths: str,
                 drug_preprocess_type: Tuple[str, Dict] | List[Tuple[str, Dict] | None] | None,
                 protein_preprocess_type: Tuple[str, Dict] | List[Tuple[str, Dict] | None] | None,
                 protein_attributes: List[str] | str | None,
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

        super().__init__(file_paths, drug_preprocess_type, drug_attributes, online_preprocessing_drug, in_memory_preprocessing_drug, protein_preprocess_type, protein_attributes, online_preprocessing_protein,
                         in_memory_preprocessing_protein, label_attributes, label_preprocess_type, online_preprocessing_label, in_memory_preprocessing_label, save_directory, urls, common_columns, separators, associated_model, threads)

        save_directory = self.file_paths if save_directory is None else save_directory

        if not os.path.exists(celegans_data_path) or not os.path.exists(celegansSeqPdb_path):
            self.download()
