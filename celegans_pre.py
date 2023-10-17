from random import shuffle

import pandas as pd

from deepdrugdomain.data import DrugProteinDataset
import resource


df = pd.read_csv("data/drugbank/drugbankSeqPdb.txt")
with open("data/drugbank/DrugBank.txt", 'r') as fp:
    train_raw = fp.read()
raw_data = train_raw.split("\n")
shuffle(raw_data)
train_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
for item in raw_data:
    try:
        a = item.split()
        smile = a[2]
        sequence = a[3]
        pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
        if pdb_code is not None:
            label = 1 if a[4] == '1' else 0
            train_df = train_df.append({'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label},
                            ignore_index=True)
    except:
        pass

train_df = train_df.head(100)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))

dataset = DrugProteinDataset(
    train_df, drug_preprocess_type=("dgl_graph_from_smile_fragments", {"fragment": True, "max_block": 6, "max_sr": 8, "min_frag_atom": 1}),
    drug_attributes="SMILE",
    online_preprocessing_drug=True,
    protein_preprocess_type=("dgl_graph_from_pocket", {"pdb_path": "data/pdb/", "protein_size_limit": 50000}),
    protein_attributes="PDB",
    online_preprocessing_protein=False,
    shard_directory="data/celegans/",
    threads=4
)
