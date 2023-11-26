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

### Documentation
For now please read the docstring inside the module for more information.

### Contributing
We welcome contributions to DeepDrugDomain! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for more details on how to contribute.

### Citation
We don't have a paper yet!
