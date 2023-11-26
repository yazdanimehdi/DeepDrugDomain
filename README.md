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

| Preprocessing Method | Converts Data From | Converts Data To | Settings                                  | Used in Models    |
| -------------------- | ------------------ | ---------------- | ----------------------------------------- | ----------------- |
| Preprocess1          | Data Format 1      | Data Format A    | Setting1: value, Setting2: value          | Model1, Model2    |
| Preprocess2          | Data Format 2      | Data Format B    | Setting3: range, Setting4: [option1, ...] | Model3            |
| Preprocess3          | Data Format 3      | Data Format C    | Setting5: value                           | Model2, Model4    |
| ...                  | ...                | ...              | ...                                       | ...               |

### Usage Example
```python
import deepdrugdomain as ddd
from dgllife.utils import CanonicalAtomFeaturizer

feat = CanonicalAtomFeaturizer() 
preprocess_drug = ddd.data.PreprocessingObject(attribute="SMILES", preprocessing_type="smile_to_dgl_graph", preprocessing_settings={
                                               "fragment": False, "node_featurizer": feat}, in_memory=True, online=False)
```

## Supported Datasets

The following table provides information about the datasets supported by our package:

| Dataset Name  | Description                                      | Use Case                   |
| ------------- | ------------------------------------------------ | -------------------------- |
| Dataset1      | Brief description of what Dataset1 consists of. | Use case of Dataset1.      |
| Dataset2      | Brief description of what Dataset2 consists of. | Use case of Dataset2.      |
| Dataset3      | Brief description of what Dataset3 consists of. | Use case of Dataset3.      |
| ...           | ...                                              | ...                        |

### Supported Split Methods

All datasets listed above support the following split methods:
- Method1 (e.g., Random Split)
- Method2 (e.g., Stratified Split)
- Method3 (e.g., Time-based Split)


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

The following table showcases the models supported by our package and the datasets each model is compatible with:

| Model          | Supported Datasets          |
| -------------- | --------------------------- |
| Model1         | Dataset1, Dataset2          |
| Model2         | Dataset3, Dataset4, Dataset5|
| Model3         | Dataset6                    |
| ...            | ...                         |



## Documentation
For now please read the docstring inside the module for more information.

## Contributing
We welcome contributions to DeepDrugDomain! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for more details on how to contribute.

## Citation
We don't have a paper yet!
