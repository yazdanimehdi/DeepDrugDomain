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
dataset = ddd.data.DatasetFactory.create("drugbank",
                                    file_paths="data/drugbank/",
                                     drug_preprocess_type=("dgl_graph_from_smile",
                                     {"fragment": False, "max_block": 6, "max_sr": 8, "min_frag_atom": 1}),
                                     drug_attributes="SMILE",
                                    online_preprocessing_drug=False,
                                    in_memory_preprocessing_drug=True,
                                    protein_preprocess_type=(
                                        "dgl_graph_from_protein_pocket", {"pdb_path": "data/pdb/", "protein_size_limit": 10000}),
                                    protein_attributes="pdb_id",
                                    online_preprocessing_protein=False,
                                    in_memory_preprocessing_protein=False,
                                    label_attributes="Label",
                                    save_directory="data/drugbank/",
                                    threads=8
    )

datasets = dataset(split_method="random_split", frac=[0.8, 0.1, 0.1])
collate_fn = ddd.data.CollateFactory.create("binding_graph_smile_graph")
data_loader_train = DataLoader(datasets[0], batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=collate_fn, drop_last=True)

data_loader_val = DataLoader(datasets[1], drop_last=False, batch_size=32,
                              num_workers=4, pin_memory=False, collate_fn=collate_fn)
data_loader_test = DataLoader(datasets[2], drop_last=False, batch_size=32, collate_fn=collate_fn,
                              num_workers=4, pin_memory=False)
model = ddd.models.ModelFactory.create("attentionsitedti")
optimizer = ddd.optimizers.OptimizerFactory.create(
    "adamw", model.parameters(), lr=1e-4, weight_decay=0.03)
scheduler = ddd.schedulers.SchedulerFactory.create("cosine", optimizer)
device = torch.device("cpu")
model.to(device)
```

### Documentation
For now please read the docstring inside the module for more information.

### Contributing
We welcome contributions to DeepDrugDomain! Please check out our [Contribution Guidelines](CONTRIBUTE.md) for more details on how to contribute.

### Citation
We don't have a paper yet!
