from ..base_dataset import DatasetFactory, DatasetInfo
from ..default_dataset import DrugProteinDataset
import pandas as pd

# Assuming custom processing function is needed, define it here:


def drugbank_processing_function(df: pd.DataFrame) -> pd.DataFrame:
    # Insert custom processing logic for DrugBank dataset here.
    # For example, let's concatenate SMILE and sequence columns:
    df['combined_info'] = df['SMILE'].astype(
        str) + "_" + df['sequence'].astype(str)
    return df


# Register the DrugBank dataset class with associated files and processing information.
@DatasetFactory.register(
    key="drugbank_dti",
    file_paths=["path/to/drugbankSeqPdb.csv", "path/to/DrugBank.csv"],
    common_columns={"sequence": "TargetSequence"},
    default_columns={
        'SMILE': str,
        'sequence': str,
        'PDB': str,
        'Label': int,
        'drug_id': str,  # Additional columns specific to DrugBank
        # ... more columns as needed
    },
    separator=",",  # CSV file separator
    processing_function=drugbank_processing_function
)
class DrugBankDataset:
    # Additional methods and attributes specific to DrugBank can be added here.
    pass
