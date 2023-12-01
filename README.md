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
conda create --name deepdrugdomain python=3.11
conda activate deepdrugdomain
conda install -c conda-forge rdkit
pip install git+https://github.com/yazdanimehdi/deepdrugdomain.git
```

## Quick Start

```python
import deepdrugdomain as ddd

data_loaders, model, optimizer, criterion, scheduler, evaluators = ddd.utils.initialize_training_environment(model="supported_model_name", dataset="supported_dataset_by_the_model")

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
for i for range(epochs):
    print(f"Epoch {i}:")
    # train
    model.train_one_epoch(data_loaders[0], device, criterion, optimizer, num_epochs=epochs, scheduler=scheduler, evaluator=evaluator[0])
    # validation
    print(model.evaluate(data_loaders[1], device, criterion, evaluator=evaluator[1])) 

# testing the trained model
model.evaluate(data_loaders[2], device, criterion, evaluator=evaluator[1]) 
```
## Examples

The [example](./examples/) folder contains a collection of scripts and notebooks demonstrating various capabilities of DeepDrugDomain. Below is an overview of what each example covers:

### Training Different Models

- [attentionsitedti.ipynb](./examples/attentionsitedti.ipynb): Brief explanation of training AttentionSiteDTI with custom configurations and model tampering in this Jupyter Notebook.

### Other Functionalities

## Supported Preprocessings

The following table lists the preprocessing methods supported by the package, detailing the data conversion, settings options, and the models that use them:

Ligand:

| Preprocessing Method | Converts Data From | Converts Data To | Settings                                  | Used in Models    |
| -------------------- | ------------------ | ---------------- | ----------------------------------------- | ----------------- |
| smiles_to_encoding          | smile     | encoding_tensor   | one_hot: bool, embedding_dim: Optional[int], max_sequence_length: Optional[int], replacement_dict: Dict[str, str], token_regex: Optional[str] , from_set: Optional[Dict[str, int]] | DrugVQA, AttentionDTA |
| smile_to_graph          | smile    | graph    | node_featurizer: Callable, edge_featurizer: Optional[Callable], consider_hydrogen: bool, fragment: bool, hops: int | AMMVF, AttentionSiteDTI, FragXsiteDTI, CSDTI            |
| smile_to_fingerprint          | smile     | fingerprint    | methods: str (['rdkit', 'morgan', 'daylight','ErG', 'rdkit2d', 'pubchem', 'ammvf', 'custom']),  radius: Optional[int], nBits: Optional[int], num_finger: Optional[int], atom_dict: Optional[AtomDictType], bond_dict: Optional[BondDictType], fingerprint_dict: Optional[FingerprintDictType],  edge_dict: Optional[Dict], consider_hydrogen: bool, custom_fingerprint: Optional[Callable],                         | AMMVF    |



Protein:

| Preprocessing Method | Converts Data From | Converts Data To | Settings                                  | Used in Models    |
| -------------------- | ------------------ | ---------------- | ----------------------------------------- | ----------------- |
| contact_map_from_pdb          | pdb_id      | contact_map    | pdb_path: str, method: str, distance_threshold: float, normalize_distance: bool| DrugVQA |
| sequence_to_fingerprint          | protein_sequence     | fingerprint    | method: str (['quasi', 'aac', 'paac', 'ct', 'custom']) | DrugVQA-Sequence            |
| kmers          | protein_sequence    | kmers_encoded_tensor   | ngram: int, word_dict: Optional[dict], max_length: Optional[int]                          | AMMVF, CSDTI   |
| protein_pockets_to_dgl_graph                  | pdb_id                | binding_pocket_graph              | pdb_path: str, protein_size_limit: int                                       | AttentionSiteDTI, FragXsiteDTI             |
|word2vec|protein_sequence|word2vec_tensor|model_path: str, vec_size: int, k: int, update_vocab: Optional[bool]| AMMVF
|sequence_to_one_hot| protein_sequence|encoding_tensor| amino_acids: str, max_sequence_length: Optional[int], one_hot: bool| AttentionDTA

Label:

| Preprocessing Method | Converts Data From | Converts Data To | Settings                                  
| -------------------- | ------------------ | ---------------- | ----------------------------------------- 
| interaction_to_binary          | binary     | binary_tensor   |          []
| ic50_to_binary          | ic50      | binary  | threshold: float
| Kd_to_binary          | kd     | binary   | threshold: float                          
| value_to_log       | float        | log           |                        []            


### Usage Example
```python
import deepdrugdomain as ddd
from dgllife.utils import CanonicalAtomFeaturizer

feat = CanonicalAtomFeaturizer() 
preprocess_drug = ddd.data.PreprocessingObject(attribute="SMILES", from_dtype="smile", to_dtype="graph", preprocessing_settings={
                                               "fragment": False, "node_featurizer": feat}, in_memory=True, online=False)
```

## Supported Datasets

The following table provides information about the datasets supported by our package:

| Dataset Name  | Description                                      | Use Case                   |
| ------------- | ------------------------------------------------ | -------------------------- |
| Celegans      | ... | DTI      |
| Human      | Brief description of what Dataset2 consists of. | DTI    |
| DrugBankDTI      | Brief description of what Dataset3 consists of. | DTI     |
| Kiba           | ...                                              | DTA, DTI                        |
| Davis           | ...                                              | DTA, DTI                        |
| IBM_BindingDB           | ...                                              | DTA, DTI                        |
| BindingDB           | ...                                              | DTA, DTI                        |
| DrugTargetCommon| ... | DTA, DTI|
| All TDC Datasets           | ...                                              | All drug discovery Tasks                        |

### Supported Split Methods

All datasets listed above support the following split methods:
- k_fold 
- random_split
- cold_split
- scaffold_split

### Usage Example
```python
import deepdrugdomain as ddd

# Define PreprocessorObject
preprocess = ...

# Load dataset
dataset = ddd.data.DatasetFactory.create("human", file_paths="data/human/", preprocesses=preprocesses) 
datasets = dataset(split_method="random_split"), frac=[0.8, 0.1, 0.1], seed=4)
```

## Supported Models and Datasets
**Disclaimer**: This implementation of DeepDrugDomain is not an official version and may contain inaccuracies or differences compared to the original models. While efforts have been made to ensure reliability, the models provided may not perform at the same level as officially published versions and should be used with this understanding.

The following table showcases the models supported by our package and the datasets each model is compatible with:

| Model          | Supported Datasets          |
| -------------- | --------------------------- |
| AttentionSiteDTI         | DTI, DTA          |
| FragXsiteDTI         | DTI, DTA   |
| DrugVQA         | DTI, DTA                    |
| CSDTI     | DTI, DTA                            |
| AMMVF | DTI, DTA   |
| AttentionDTA| DTI, DTA    |
| DeepDTA| DTI, DTA|
| WideDTA|DTI, DTA   |
| GraphDTA| DTI, DTA|
| DGraphDTA| DTI, DTA|

**Contribution**: We are actively looking to add new models to the package. Feel free to add any model to the package and shoot a pull request!

## Documentation
For now please read the docstring inside the module for more information.

## Contributing
We welcome contributions to DeepDrugDomain! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for more details on how to contribute.

## Citation
We don't have a paper yet!
