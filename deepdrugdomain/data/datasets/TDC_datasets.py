import os
from typing import Dict, List, Optional, Tuple, Union
from deepdrugdomain.data.utils import CustomDataset
from deepdrugdomain.data.utils import DatasetFactory
try:
    from tdc.multi_pred import DTI, DDI, PPI, GDA, MTI, PeptideMHC, TCREpitopeBinding, TrialOutcome, Catalyst, AntibodyAff, DrugRes, DrugSyn
    from tdc.single_pred import ADME, Tox, CRISPROutcome, Develop, Paratope, Epitope, HTS, QM, Yields
    from tdc.generation import MolGen, Reaction, RetroSyn, SBDD
    from tdc.metadata import *
except ImportError:
    raise ImportError("Please install the PyTDC package to use this dataset")


class TDCBaseDataset(CustomDataset):
    """
    Base class for Therapeutics Data Commons (TDC) datasets.

    This class serves as a base for various dataset types provided by the TDC package.
    It ensures that the given dataset name is supported for the specified dataset type
    and loads the dataset into a DataFrame.

    Attributes:
        file_paths (str): Path to the dataset files.
        data_name (str): Name of the dataset.
        df (DataFrame): Loaded dataset.

    Args:
        file_paths (str): Path to the dataset files.
        data_name (str): Name of the dataset.
        dataset_class: Corresponding dataset class from the TDC package.
        supported_datasets (list): List of supported dataset names for the dataset class.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, file_paths: str, data_name: str, dataset_class, supported_datasets, **kwargs):
        assert data_name in supported_datasets, f"Data name {data_name} not supported for {dataset_class.__name__}"
        self.file_paths = file_paths
        self.data_name = data_name
        df = dataset_class(
            path=file_paths, name=data_name).get_data(format='df')
        super().__init__(file_paths, df=df, **kwargs)


@DatasetFactory.register('TDC_DTI')
class TDCDTI(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = dti_dataset_names
        super().__init__(file_paths, data_name, DTI, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_DDI')
class TDCDDI(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = ddi_dataset_names
        super().__init__(file_paths, data_name, DDI, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_PPI')
class TDCPPI(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = ppi_dataset_names
        super().__init__(file_paths, data_name, PPI, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_ADME')
class TDCADME(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = adme_dataset_names
        super().__init__(file_paths, data_name, ADME, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Tox')
class TDCTox(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = toxicity_dataset_names
        super().__init__(file_paths, data_name, Tox, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_AntibodyAff')
class TDCAntibodyAff(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = antibodyaff_dataset_names
        super().__init__(file_paths, data_name, AntibodyAff, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_DrugRes')
class TDCDrugRes(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = drugres_dataset_names
        super().__init__(file_paths, data_name, DrugRes, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_DrugSyn')
class TDCDrugSyn(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = drugsyn_dataset_names
        super().__init__(file_paths, data_name, DrugSyn, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Catalyst')
class TDCCatalyst(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = catalyst_dataset_names
        super().__init__(file_paths, data_name, Catalyst, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_GDA')
class TDCGDA(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = gda_dataset_names
        super().__init__(file_paths, data_name, GDA, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_MTI')
class TDCMTI(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = mti_dataset_names
        super().__init__(file_paths, data_name, MTI, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_PeptideMHC')
class TDCPeptideMHC(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = peptidemhc_dataset_names
        super().__init__(file_paths, data_name, PeptideMHC, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_TCREpitopeBinding')
class TDCTCREpitopeBinding(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = tcr_epi_dataset_names
        super().__init__(file_paths, data_name,
                         TCREpitopeBinding, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_TrialOutcome')
class TDCTrialOutcome(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = trial_outcome_dataset_names
        super().__init__(file_paths, data_name, TrialOutcome, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Develop')
class TDCDevelop(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = develop_dataset_names
        super().__init__(file_paths, data_name, Develop, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_CrisprOutcome')
class TDCCrisprOutcome(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = crisproutcome_dataset_names
        super().__init__(file_paths, data_name, CRISPROutcome, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Paratope')
class TDCParatope(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = paratope_dataset_names
        super().__init__(file_paths, data_name, Paratope, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Epitope')
class TDCEpitope(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = epitope_dataset_names
        super().__init__(file_paths, data_name, Epitope, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_HTS')
class TDCHTS(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = hts_dataset_names
        super().__init__(file_paths, data_name, HTS, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_QM')
class TDCQM(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = qm_dataset_names
        super().__init__(file_paths, data_name, QM, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Yields')
class TDCYields(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = yield_dataset_names
        super().__init__(file_paths, data_name, Yields, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_MolGen')
class TDCMolGen(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = single_molecule_dataset_names
        super().__init__(file_paths, data_name, MolGen, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_Reaction')
class TDCReaction(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = forwardsyn_dataset_names
        super().__init__(file_paths, data_name, Reaction, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_RetroSyn')
class TDCRetroSyn(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = retrosyn_dataset_names
        super().__init__(file_paths, data_name, RetroSyn, supported_datasets, **kwargs)


@DatasetFactory.register('TDC_SBDD')
class TDCSBDD(TDCBaseDataset):
    def __init__(self, file_paths: str, data_name: str, **kwargs):
        supported_datasets = multiple_molecule_dataset_names
        super().__init__(file_paths, data_name, SBDD, supported_datasets, **kwargs)
