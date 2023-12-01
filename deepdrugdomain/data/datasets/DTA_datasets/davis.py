import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from deepdrugdomain.data.preprocessing.utils.preprocessing_data_struct import PreprocessingObject
from ..factory import DatasetFactory


@DatasetFactory.register('davis')
class Davis(CustomDataset):
    """
    """

    def __init__(self, file_paths: str,
                 preprocesses: PreprocessingObject,
                 save_directory: str | None = None,
                 urls: List[str] | str | None = ['https://github.com/yazdanimehdi/DeepDrugDomain/raw/main/data/davis/',
                                                 'https://github.com/yazdanimehdi/DeepDrugDomain/raw/main/data/celegans/celegansSeqPdb.txt'],
                 common_columns: Dict[str,
                                      str] | List[Dict[str, str]] | None = {'sequence': 'Target_Seq'},
                 separators: List[str] | str = [','],
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
