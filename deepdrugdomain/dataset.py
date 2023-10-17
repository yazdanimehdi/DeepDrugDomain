import os
from random import shuffle

import dgl
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from .utils import process_protein, process_smile_graph, integer_label_protein, node_featurizer
from tqdm import tqdm
from Bio.PDB import PDBList
import pickle
import subprocess
from signal import signal, SIGSEGV
from dgllife.utils import smiles_to_bigraph


class DTIData(Dataset):
    def __init__(self, name, df_dir, processed_file_dir, pdb_dir, p_graph, s_graph):
        super().__init__()
        self.p_graph = p_graph
        self.s_graph = s_graph
        self.name = name
        self.df_dir = df_dir
        self.df = pd.read_csv(df_dir)
        self.pdb_dir = pdb_dir
        self.processed_file_dir = processed_file_dir + self.name + '.pkl'
        if not os.path.exists(processed_file_dir):
            os.mkdir(processed_file_dir)
        if not os.path.exists(self.processed_file_dir):
            self.p_graph = {}
            self.s_graph = {}
            self.pre_process()
        else:
            self.df = self.df[self.df['PDB'].isin(self.p_graph.keys())]
            self.df = self.df[self.df['SMILE'].isin(self.s_graph.keys())]

    def pre_process(self):
        not_available_smile = []
        not_available_pdb = []
        for i in tqdm(self.df['PDB'].unique()):
            pdb = i.lower()
            if pdb not in self.p_graph.keys():
                try:

                    if not os.path.exists(self.pdb_dir + pdb + '.pdb'):
                        pdbl = PDBList(verbose=False)
                        pdbl.retrieve_pdb_file(
                            pdb, pdir=self.pdb_dir, overwrite=False, file_format="pdb"
                        )
                        # Rename file to .pdb from .ent
                        os.rename(
                            self.pdb_dir + "pdb" + pdb + ".ent", self.pdb_dir + pdb + ".pdb"
                        )
                        # Assert file has been downloaded
                        assert any(pdb in s for s in os.listdir(self.pdb_dir))

                    constructed_graphs = process_protein(self.pdb_dir + pdb + ".pdb")
                    self.p_graph[pdb] = constructed_graphs

                except Exception as e:
                    not_available_pdb.append(pdb)

        for smile in tqdm(self.df['SMILE'].unique()):
            if smile not in self.s_graph:
                try:
                    self.s_graph[smile] = process_smile_graph(smile, 6, 8, 1)
                except Exception as e:
                    not_available_smile.append(smile)

        not_available = []
        for i in range(len(self.df.index)):
            pdb = self.df.iloc[i]['PDB'].lower()
            smile = self.df.iloc[i]['SMILE']
            if pdb in not_available_pdb or smile in not_available_smile:
                not_available.append(i)

        self.df.drop(list(set(not_available)), axis=0, inplace=True)
        self.df.to_csv(self.df_dir)
        with open(self.processed_file_dir, 'wb') as fp:
            pickle.dump([self.p_graph, self.s_graph], fp)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        smile = self.df.iloc[index]['SMILE']
        pdb = self.df.iloc[index]['PDB'].lower()
        p_graph = self.p_graph[pdb]
        # s_graph = self.s_graph[smile]
        g = smiles_to_bigraph(smile, node_featurizer=node_featurizer)
        g = dgl.add_self_loop(g)
        y = torch.tensor(self.df.iloc[index]['Label'])
        return p_graph, g, y


def collate_wrapper(batch):
    transposed_data = list(zip(*batch))
    prot_graph = transposed_data[0]
    target_graph = transposed_data[1]
    inp = prot_graph, target_graph
    tgt = torch.stack(transposed_data[2], 0)
    return inp, tgt
