import argparse
import numpy as np
import torch
from deepdrugdomain.optimizers.factory import OptimizerFactory
from deepdrugdomain.schedulers.factory import SchedulerFactory
from torch.utils.data import DataLoader
from deepdrugdomain.models.factory import ModelFactory
from deepdrugdomain.utils.config import args_to_config
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
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
    edge_feat = CanonicalBondFeaturizer()
    # preprocess_drug = ddd.data.PreprocessingObject(attribute="SMILES", preprocessing_type="smile_to_dgl_graph", preprocessing_settings={
    #                                                "fragment": True, "max_block": 6, "max_sr": 8, "min_frag_atom": 1, "node_featurizer": feat}, in_memory=True, online=False)
    # preprocess_protein = ddd.data.PreprocessingObject(attribute="pdb_id", preprocessing_type="protein_pockets_to_dgl_graph", preprocessing_settings={
    #                                                   "pdb_path": "data/pdb/", "protein_size_limit": 10000}, in_memory=False, online=False)
    # preprocess_label = ddd.data.PreprocessingObject(
    #     attribute="Label", preprocessing_type="interaction_to_binary", preprocessing_settings={}, in_memory=True, online=True)

    # preprocess_drug = ddd.data.PreprocessingObject(attribute="SMILES", preprocessing_type="smiles_to_embedding", preprocessing_settings={
    #     "max_sequence_length": 247}, in_memory=True, online=False)
    # preprocess_protein = ddd.data.PreprocessingObject(
    #     attribute="pdb_id", preprocessing_type="contact_map_from_pdb", preprocessing_settings={"pdb_path": "data/pdb/"}, in_memory=False, online=False)
    # preprocess_label = ddd.data.PreprocessingObject(
    #     attribute="Label", preprocessing_type="interaction_to_binary", preprocessing_settings={}, in_memory=True, online=True)

    # preprocess_drug1 = ddd.data.PreprocessingObject(attribute="SMILES", preprocessing_type="smile_to_dgl_graph", preprocessing_settings={
    #     "fragment": False, "node_featurizer":  ddd.data.preprocessing.ammvf_mol_features, "consider_hydrogen": True}, in_memory=True, online=False)

    # preprocess_drug2 = ddd.data.PreprocessingObject(attribute="SMILES", preprocessing_type="smile_to_fingerprint", preprocessing_settings={
    #                                                 "method": "ammvf", "consider_hydrogen": True}, in_memory=True, online=False)

    # preprocess_protein1 = ddd.data.PreprocessingObject(attribute="Target_Seq", preprocessing_type="word2vec", preprocessing_settings={
    #                                                    "model_path": "data/human/word2vec.model", "vec_size": 100}, in_memory=True, online=False)
    # preprocess_protein2 = ddd.data.PreprocessingObject(
    #     attribute="Target_Seq", preprocessing_type="kmers", preprocessing_settings={"ngram": 3}, in_memory=True, online=False)

    # preprocess_label = ddd.data.PreprocessingObject(
    #     attribute="Label", preprocessing_type="interaction_to_binary", preprocessing_settings={}, in_memory=True, online=True)

    # preprocesses = preprocess_drug1 + preprocess_drug2 + \
    #     preprocess_protein1 + preprocess_protein2 + preprocess_label

    # preprocess_drug1 = ddd.data.PreprocessingObject(attribute="SMILES", from_dtype="smile", to_dtype="graph", preprocessing_settings={
    #     "fragment": False, "node_featurizer":  feat, "consider_hydrogen": False, "consider_hydrogen": True}, in_memory=True, online=False)

    # preprocess_drug2 = ddd.data.PreprocessingObject(attribute="SMILES", from_dtype="smile", to_dtype="graph", preprocessing_settings={
    #     "fragment": False, "node_featurizer":  feat, "consider_hydrogen": False, "hops": 2, "consider_hydrogen": True}, in_memory=True, online=False)

    # preprocess_protein = ddd.data.PreprocessingObject(
    #     attribute="Target_Seq", from_dtype="protein_sequence", to_dtype="kmers_encoded_tensor", preprocessing_settings={"ngram": 1, "max_length": 1200}, in_memory=True, online=False)

    # preprocess_label = ddd.data.PreprocessingObject(
    #     attribute="Label",  from_dtype="binary", to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)

    # preprocesses = preprocess_drug1 + preprocess_drug2 + preprocess_protein + preprocess_label
    model = ModelFactory.create("drugvqa")

    preprocesses = ddd.data.PreprocessingList(
        model.default_preprocess("SMILES", "pdb_id", "Label"))
    # preprocesses = preprocess_drug + preprocess_protein + preprocess_label
    dataset = ddd.data.DatasetFactory.create(
        "human", file_paths="data/human/", preprocesses=preprocesses)
    datasets = dataset(split_method="random_split",
                       frac=[0.8, 0.1, 0.1], seed=seed, sample=0.1)
    collate_fn = model.collate

    data_loader_train = DataLoader(
        datasets[0], batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    data_loader_val = DataLoader(datasets[1], drop_last=False, batch_size=1,
                                 num_workers=4, pin_memory=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(datasets[2], drop_last=False, batch_size=1,
                                  num_workers=4, pin_memory=False, collate_fn=collate_fn)
    criterion = torch.nn.BCELoss()
    optimizer = OptimizerFactory.create(
        "adam", model.parameters(), lr=0.0005, weight_decay=0.0)
    scheduler = SchedulerFactory.create(
        "cosine", optimizer, warmup_epochs=0, warmup_lr=1e-3, num_epochs=200)
    device = torch.device("cpu")
    model.to(device)
    train_evaluator = ddd.metrics.Evaluator(["accuracy_score"], threshold=0.5)
    test_evaluator = ddd.metrics.Evaluator(
        ["accuracy_score", "f1_score", "auc", "precision_score", "recall_score"], threshold=0.5)
    epochs = 3000
    accum_iter = 1
    print(model.evaluate(data_loader_val, device,
          criterion, evaluator=test_evaluator))
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        model.train_one_epoch(data_loader_train, device, criterion,
                              optimizer, num_epochs=200, scheduler=scheduler, evaluator=train_evaluator, grad_accum_steps=accum_iter)
        print(model.evaluate(data_loader_val, device,
                             criterion, evaluator=test_evaluator))

    print(model.evaluate(data_loader_test, device,
                         criterion, evaluator=test_evaluator))

    # scheduler.step()
    # test_func(model, data_loader_val, device)
    # test_func(model, data_loader_test, device)
    # fn = "last_checkpoint_celegans.pt"
    # info_dict = {
    #     'epoch': epoch,
    #     'net_state': model.state_dict(),
    #     'optimizer_state': optimizer.state_dict()
    # }
    # torch.save(info_dict, fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DTIA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
