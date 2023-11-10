import argparse
from random import shuffle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
from torch.optim.lr_scheduler import ExponentialLR

from deepdrugdomain.data import CustomDataset
from deepdrugdomain.data.collate import CollateFactory
from torch.utils.data import DataLoader
from deepdrugdomain.models.factory import ModelFactory
from deepdrugdomain.utils.config import args_to_config
from pathlib import Path
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(
        'DTIA training and evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--raw-data-dir', default='./data/', type=str)
    parser.add_argument('--train-split', default=1, type=float)
    parser.add_argument('--val-split', default=0, type=float)
    parser.add_argument('--dataset', default='drugbank',
                        choices=['dude', 'celegans', 'human', 'drugbank',
                                 'ibm', 'bindingdb', 'kiba', 'davis'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--df-dir', default='./data/', type=str)
    parser.add_argument('--processed-file-dir',
                        default='./data/processed/', type=str)
    parser.add_argument('--pdb-dir', default='./data/pdb/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    config = args_to_config(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = CustomDataset(
        file_paths=["data/drugbank/DrugBank.txt",
                    "data/drugbank/drugbankSeqPdb.txt"],
        common_columns={"sequence": "TargetSequence"},
        separators=[" ", ","],
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
    dataset_train, dataset_val, dataset_test = dataset([0.8, 0.1, 0.1])
    collate_fn = CollateFactory.create("binding_graph_smile_graph")
    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                                   collate_fn=collate_fn, drop_last=True)

    data_loader_val = DataLoader(dataset_val, drop_last=False, batch_size=32,
                                 num_workers=4, pin_memory=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, drop_last=False, batch_size=32, collate_fn=collate_fn,
                                  num_workers=4, pin_memory=False)
    model = ModelFactory.create("AMMVF")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.03)
    criterion = torch.nn.BCELoss()
    scheduler = ExponentialLR(optimizer, gamma=0.98)
    device = torch.device("cpu")
    model.to(device)
    epochs = 200
#     accum_iter = 2
#     for epoch in range(epochs):
#         losses = []
#         accs = []
#         model.train()
#         with tqdm(data_loader_train) as tepoch:
#             tepoch.set_description(f"Epoch {epoch}")
#             for batch_idx, (inp, target) in enumerate(tepoch):
#                 outs = []
#                 for item in range(len(inp[0])):
#                     inpu = (inp[0][item].to(device), inp[1][item].to(device))
#                     out = model(inpu)
#                     outs.append(out)
#                 out = torch.stack(outs, dim=0).squeeze(1)

#                 target = target.to(device).view(-1, 1).to(torch.float)
#                 loss = criterion(out, target)
#                 matches = [torch.round(i) == torch.round(j)
#                            for i, j in zip(out, target)]
#                 acc = matches.count(True) / len(matches)
#                 accs.append(acc)
#                 losses.append(loss.detach().cpu())
#                 loss.backward()
#                 if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader_train)):
#                     optimizer.step()
#                     optimizer.zero_grad()
#                 acc_mean = np.array(accs).mean()
#                 loss_mean = np.array(losses).mean()
#                 tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)
#         scheduler.step()
#         test_func(model, data_loader_val, device)
#         test_func(model, data_loader_test, device)
#         fn = "last_checkpoint_celegans.pt"
#         info_dict = {
#             'epoch': epoch,
#             'net_state': model.state_dict(),
#             'optimizer_state': optimizer.state_dict()
#         }
#         torch.save(info_dict, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DTIA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
