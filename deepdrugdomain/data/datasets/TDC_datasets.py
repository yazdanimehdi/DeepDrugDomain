import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from deepdrugdomain.data.utils import DatasetFactory
try:
    from tdc.multi_pred import DTI
except ImportError:
    raise ImportError("Please install the PyTDC package to use this dataset")


@DatasetFactory.register('TDC_DTI')
class TDC(CustomDataset):
    def __init__(self, file_paths: str,
                 data_name: str,
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
                 urls: List[str] | str | None = None,
                 common_columns: Dict[str,
                                      str] | List[Dict[str, str]] | None = None,
                 separators: List[str] | str | None = None,
                 associated_model: str | None = None,
                 threads: int = 4) -> None:

        assert data_name in ['davis',
                             'kiba',
                             'bindingdb_kd',
                             'bindingdb_ic50',
                             'bindingdb_ki',
                             'bindingdb_patent'], f"Data name {data_name} not supported for TDC_DTI"
        self.file_paths = file_paths
        self.data_name = data_name
        df = DTI(path=file_paths, name=data_name).get_data(format='df')
        super().__init__(file_paths, drug_preprocess_type, drug_attributes, online_preprocessing_drug, in_memory_preprocessing_drug, protein_preprocess_type, protein_attributes, online_preprocessing_protein,
                         in_memory_preprocessing_protein, label_attributes, label_preprocess_type, online_preprocessing_label, in_memory_preprocessing_label, save_directory, urls, common_columns, separators, associated_model, threads, df=df)
