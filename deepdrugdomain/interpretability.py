import torch

from deepdrugdomain.data.preprocessing.drug import macfrag
from dataset import collate_wrapper
from build_dataset import build_dataset
from torch.utils.data import DataLoader
from deepdrugdomain.utils.config import Config
from deepdrugdomain.models import PerceiverIODTI

from rdkit import Chem

config = {'dataset': "human", "processed_file_dir": "./data/processed/", "raw_data_dir": "./data/",
          "df_dir": "./data/", "train_split": 0.8, "val_split": 0.2, "pdb_dir": "./data/pdb"}

config = Config(**config)
dataset_train, dataset_val, dataset_test = build_dataset(config=config)
data_loader_val = DataLoader(dataset_val, drop_last=False, batch_size=1, collate_fn=collate_wrapper,
                              num_workers=0, pin_memory=False)

model = PerceiverIODTI()
a = torch.load("../last_checkpoint.pt")['net_state']
model.load_state_dict(a)
device = torch.device(0)
model.to(device)
for idx, (inp, target) in list(enumerate(data_loader_val))[0: 1]:
    outs = []
    for item in range(len(inp[0])):
        inpu = (inp[0][item].to(device), inp[1][item].to(device))
        smile = inp[2][0]
        pdb = inp[3][0]
        out, binding_attn, frag_attn = model(inpu)
        print(out)
        print(target)

mol = Chem.MolFromSmiles(smile)
frags = macfrag.MacFrag(mol, maxBlocks=6, maxSR=8, asMols=False, minFragAtoms=1)


def get_attention_map(att_mat):
    att_mat = torch.stack(att_mat).squeeze(1)

    att_mat = torch.mean(att_mat, dim=1)
    joint_attentions = 1
    for n in range(0, att_mat.size(0)):
        joint_attentions = torch.mean(att_mat[n], dim=0) * joint_attentions

    v = joint_attentions
    return v

frag_prob = torch.sum(get_attention_map(frag_attn).view(2, -1), dim=0)
site_prob = get_attention_map(binding_attn)