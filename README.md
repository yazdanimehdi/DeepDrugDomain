# DeepDrugDomain

DeepDrugDomain is a comprehensive Python toolkit aimed at simplifying and accelerating the process of drug-target interaction (DTI) and drug-target affinity (DTA) prediction using deep learning. With a flexible preprocessing pipeline and modular design, DeepDrugDomain supports innovative research and development in computational drug discovery.

## Features

- **Extensive Preprocessing Capabilities:** Quickly prepare datasets for modeling with a wide range of built-in preprocessing functions.
- **Modular Design:** Easy-to-use factories for creating models, tasks, collate functions, and more, with just a few lines of code.
- **Stateful Evaluation Metrics:** Integrated evaluation metrics to monitor model performance and ensure reproducibility.
- **Flexible Activation Function Registry:** Register and use custom activation functions seamlessly within models.
- **Comprehensive Task Handling:** Built-in support for common tasks in drug discovery, such as DTI and DTA, with the ability to define custom tasks.
- **...**

## Installation
For now you can use this environments for usage and development,
```bash
git clone git@github.com:yazdanimehdi/DeepDrugDomain.git
conda create --name deepdrugdomain python=3.11
conda activate deepdrugdomain
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

## Quick Start

```python
import deepdrugdomain as ddd
dataset = ddd.data.DatasetFactory.create("human",
                                    file_paths="data/human/",
                                    drug_preprocess_type=[("smiles_to_embedding", {
                                                           "max_sequence_length": 247})],
                                    protein_preprocess_type=[
                                        ("contact_map_from_pdb", {
                                         "pdb_path": "data/human/pdb/"})
                                    ],
                                    protein_attributes=[
                                        "pdb_id"],
                                    in_memory_preprocessing_protein=True,
                                    drug_attributes=["SMILES"],
                                    online_preprocessing_protein=[False],)

datasets = dataset(split_method="cold_split",
                   entities="SMILES", frac=[0.8, 0.1, 0.1])

model = ddd.models.ModelFactory.create("drugvqa")
```

### Documentation
For now please read the docstring inside the module for more information.

### Contributing
We welcome contributions to DeepDrugDomain! Please check out our [Contribution Guidelines](CONTRIBUTE.md) for more details on how to contribute.

### Citation
We don't have a paper yet!
<!-- If you use DeepDrugDomain in your research, please cite:
```bibtex
@article{deepdrugdomain2024,
  title={From Data to Discovery: The DeepDrugDomain Framework for Predicting Drug-Target Interactions and Affinity},
  author={...},
  journal={...},
  year={2024}
} -->
```