import argparse
from random import shuffle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from deepdrugdomain.optimizers.factory import OptimizerFactory
from deepdrugdomain.schedulers.factory import SchedulerFactory
from deepdrugdomain.data import CustomDataset
from deepdrugdomain.data.collate import CollateFactory
from torch.utils.data import DataLoader
from deepdrugdomain.models.factory import ModelFactory
from deepdrugdomain.utils.config import args_to_config
from deepdrugdomain.data.datasets import DatasetFactory
from dgllife.utils import CanonicalAtomFeaturizer
from pathlib import Path
from tqdm import tqdm
import deepdrugdomain as ddd


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
    parser.add_argument('--seed', default=4, type=int)
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
    feat = CanonicalAtomFeaturizer()
    dataset = DatasetFactory.create("human",
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
    print(datasets[0][0])

    model = ModelFactory.create("drugvqa")
    # collate_fn = CollateFactory.create("ammvf_collate")
    data_loader_train = DataLoader(datasets[0], batch_size=32, shuffle=True, num_workers=0, pin_memory=True,
                                   drop_last=True)

    data_loader_val = DataLoader(datasets[1], drop_last=False, batch_size=32,
                                 num_workers=4, pin_memory=False)
    data_loader_test = DataLoader(datasets[2], drop_last=False, batch_size=32,
                                  num_workers=4, pin_memory=False)

    criterion = torch.nn.BCELoss()
    optimizer = OptimizerFactory.create(
        "adam", model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = SchedulerFactory.create("cosine", optimizer)
    device = torch.device("cpu")
    model.to(device)
    # model.to(torch.float64)
    epochs = 200

    for epoch in range(epochs):
        losses = []
        accs = []
        model.train()
        with tqdm(datasets[0]) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch_idx, (pic, smile, target) in enumerate(tepoch):
                outs = []
                targets = []
                smile = smile.to(device)
                smile = smile.unsqueeze(0).to(torch.long)
                pic = pic.unsqueeze(0).unsqueeze(0).to(device)
                out = torch.sigmoid(model(smile, pic))
                outs.append(out)
                target = target.to(
                    device).view(-1, 1).to(torch.float)
                targets.append(target)
                if batch_idx % 32 == 0:
                    out = torch.stack(outs, dim=0).squeeze(1)
                    target = torch.stack(targets, 0).to(
                        device).view(-1, 1).to(torch.float)
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    loss = criterion(out, target)
                    matches = [torch.round(out) == torch.round(target)]
                    acc = matches.count(True)
                    accs.append(acc)
                    losses.append(loss.detach().cpu())
                    outs = []
                    targets = []

    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    # epochs = 200
    # accum_iter = 1
    # for epoch in range(epochs):
    #     losses = []
    #     accs = []
    #     model.train()
    #     with tqdm(data_loader_train) as tepoch:
    #         tepoch.set_description(f"Epoch {epoch}")
    #         for batch_idx, (inp, target) in enumerate(tepoch):
    #             outs = []
    #             targets = []
    #             for item in range(len(inp[0])):
    #                 protein1 = inp[0][item].to(device)
    #                 protein1 = protein1.to(torch.float64)
    #                 protein2 = inp[1][item].to(device)
    #                 compound1 = inp[2][item].to(device)
    #                 compound1 = compound1.to(torch.float64)
    #                 g = inp[3][item].to(device)
    #                 out = model(protein1, protein2, compound1, g)

    #                 # out = model(inpu)
    #                 outs.append(out)
    #                 targets.append(target[item])

    #             out = torch.stack(outs, dim=0).squeeze(1)

    #             target = torch.stack(targets, 0).to(
    #                 device).view(-1).to(torch.long)
    #             loss = criterion(out, target)
    #             matches = [torch.argmax(i) == j
    #                        for i, j in zip(out, target)]
    #             acc = matches.count(True) / len(matches)
    #             accs.append(acc)
    #             losses.append(loss.detach().cpu())
    #             loss.backward()
    #             if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader_train)):
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #             acc_mean = np.array(accs).mean()
    #             loss_mean = np.array(losses).mean()
    #             tepoch.set_postfix(loss=loss_mean, accuracy=acc_mean)
    #     # scheduler.step()
    #     # test_func(model, data_loader_val, device)
    #     # test_func(model, data_loader_test, device)
    #     fn = "last_checkpoint_celegans.pt"
    #     info_dict = {
    #         'epoch': epoch,
    #         'net_state': model.state_dict(),
    #         'optimizer_state': optimizer.state_dict()
    #     }
    #     torch.save(info_dict, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DTIA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
