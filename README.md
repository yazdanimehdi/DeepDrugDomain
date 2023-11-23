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
from deepdrugdomain.models.factory import ModelFactory
from deepdrugdomain.optimizers.factory import OptimizerFactory
from deepdrugdomain.schedulers.factory import SchedulerFactory
from dgllife.utils import CanonicalAtomFeaturizer

feat = CanonicalAtomFeaturizer()
seed = 4

preprocess_drug = ddd.data.PreprocessingObject(attribute="SMILES",
                                               preprocessing_type="smile_to_dgl_graph",
                                               preprocessing_settings={"fragment": False, "node_featurizer": feat},
                                               in_memory=True,
                                               online=False)

preprocess_protein = ddd.data.PreprocessingObject(attribute="pdb_id",
                                                  preprocessing_type="protein_pockets_to_dgl_graph",
                                                  preprocessing_settings={"pdb_path": "data/pdb/", "protein_size_limit": 10000},
                                                  in_memory=False,
                                                  online=False)

preprocess_label = ddd.data.PreprocessingObject(attribute="Label",
                                                preprocessing_type="interaction_to_binary",
                                                preprocessing_settings={},
                                                in_memory=True,
                                                online=True)

preprocesses = preprocess_drug + preprocess_protein + preprocess_label

dataset = ddd.data.DatasetFactory.create("drugbank", file_paths="data/drugbank/", preprocesses=preprocesses)

datasets = dataset(split_method="random_split", frac=[0.8, 0.1, 0.1], seed=seed, sample=0.05)

collate_fn = CollateFactory.create("binding_graph_smile_graph")

data_loader_train = DataLoader(datasets[0], batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=collate_fn, drop_last=True)
data_loader_val = DataLoader(datasets[1], drop_last=False, batch_size=32,
                                num_workers=4, pin_memory=False, collate_fn=collate_fn)
data_loader_test = DataLoader(datasets[2], drop_last=False, batch_size=32, collate_fn=collate_fn,
                                num_workers=4, pin_memory=False)

model = ModelFactory.create("attentionsitedti")

criterion = torch.nn.BCELoss()

optimizer = OptimizerFactory.create("adamw", model.parameters(), lr=1e-4, weight_decay=0.03)
scheduler = SchedulerFactory.create("cosine", optimizer)

device = torch.device("cpu")
model.to(device)

```

### Documentation
For now please read the docstring inside the module for more information.

### Contributing
We welcome contributions to DeepDrugDomain! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for more details on how to contribute.

### Citation
We don't have a paper yet!
