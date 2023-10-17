"""
protein_graph_preprocessor.py

This module provides the GraphFromPocketPreprocessor class, which preprocesses protein data to produce graph representations
utilizing the Deep Graph Library (DGL). The primary focus of this preprocessor is to convert protein data (in the form of PDB files)
into graph structures based on molecular pockets.

The GraphFromPocketPreprocessor specifically implements preprocessing techniques used in the following research papers:
- AttentionSiteDTI
- BindingSiteAugmentedDTA
- FragXSiteDTI

The class utilizes the ProteinPreprocessorFactory for dynamic registration, allowing users to select this preprocessor
for their tasks dynamically.

The primary use case for this module includes, but is not limited to, preparing protein data for graph-based deep learning
models in drug discovery and bioinformatics applications.

"""

import os
from typing import Optional

import pandas as pd
from Bio.PDB import PDBList
from deepchem.dock import ConvexHullPocketFinder
from dgllife.utils import atom_type_one_hot, atom_degree_one_hot, \
    atom_implicit_valence_one_hot, atom_formal_charge, atom_num_radical_electrons, atom_hybridization_one_hot, \
    atom_is_aromatic, atom_total_num_H_one_hot, ConcatFeaturizer
from deepdrugdomain.data.data_preprocess.base_preprocessor import BasePreprocessor
from .protein_preprocessing_factory import ProteinPreprocessorFactory
from deepdrugdomain.exceptions import MissingRequiredParameterError, ProteinTooBig
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import dgl
import networkx as nx

from deepdrugdomain.data.utils import serialize_dgl_graph_hdf5, deserialize_dgl_graph_hdf5


def _atom_feature(atom):
    """
    Retrieve feature array for an atom.

    Parameters:
    - atom: An atom object.

    Returns:
    - np.array: The feature array for the given atom.
    """
    return np.array(ConcatFeaturizer([atom_type_one_hot,
                                      atom_degree_one_hot,
                                      atom_implicit_valence_one_hot,
                                      atom_formal_charge,
                                      atom_num_radical_electrons,
                                      atom_hybridization_one_hot,
                                      atom_is_aromatic,
                                      atom_total_num_H_one_hot])(atom))


def _get_atom_feature(m):
    """
    Compile feature array for each atom in the molecule.

    Parameters:
    - m: A molecule object.

    Returns:
    - np.array: A 2D array of features, one row for each atom.
    """
    h = []
    for i in range(len(m)):
        h.append(_atom_feature(m[i][0]))
    h = np.array(h)

    return h


def _get_pockets_from_protein(pdb_file: str) -> list:
    """
    Extract binding pockets from the protein.

    Parameters:
    - pdb_file (str): Path to the protein PDB file.

    Returns:
    - list: List of binding pockets.
    """
    pk = ConvexHullPocketFinder()
    pockets = pk.find_pockets(pdb_file)
    return pockets


def _get_constructed_graphs_for_pockets(pockets: list, m: Chem.Mol, am: np.ndarray, d2: np.ndarray) -> list:
    """
    Construct DGL graphs for each pocket.

    Parameters:
    - pockets (list): List of binding pockets.
    - m (Chem.Mol): The RDKit molecule object.
    - am (np.ndarray): The adjacency matrix of the molecule.
    - d2 (np.ndarray): Atom positions.

    Returns:
    - list: List of DGL graphs for each pocket.
    """
    constructed_graphs = []

    for bound_box in pockets:
        # Extract pocket dimensions
        x_min, x_max = bound_box.x_range
        y_min, y_max = bound_box.y_range
        z_min, z_max = bound_box.z_range

        # Identify atoms within this pocket
        idxs = [idx for idx, atom_cord in enumerate(d2) if
                x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max]

        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        h = _get_atom_feature([(m.GetAtoms()[i], d2[i]) for i in idxs])
        g = nx.convert_matrix.from_numpy_array(ami)
        graph = dgl.from_networkx(g)
        graph.ndata['h'] = torch.Tensor(h)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)

    return constructed_graphs


@ProteinPreprocessorFactory.register("dgl_graph_from_pocket")
class GraphFromPocketPreprocessor(BasePreprocessor):
    """
    Preprocessor to transform protein data into a graph representation using DGL.

    This class serves as a wrapper to process protein's PDB file and transform it into a DGL graph
    representation, focusing on the protein's binding pockets. The approach and representation
    utilized here are consistent with the methods cited in the following publications:
    - AttentionSiteDTI
    - BindingSiteAugmentedDTA
    - FragXSiteDTI

    Methods:
    - preprocess: Preprocesses protein data to create the DGL graph.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, data: pd.Series) -> Optional[dgl.DGLGraph]:
        """
        Preprocess the given protein PDB identifier to generate a DGL graph.

        Parameters:
        - data (pd.Series): Data row containing protein PDB identifier.
        - **kwargs: Additional parameters necessary for preprocessing.

        Returns:
        - dgl.DGLGraph: A DGL graph representation of the protein's binding pockets or None if an exception occurs.
        """
        # Convert protein PDB identifier to lowercase
        pdb = data.lower()

        # Ensure required keys are present in arguments
        required_keys = ["pdb_path", "protein_size_limit"]
        for item in required_keys:
            if item not in self.kwargs:
                raise MissingRequiredParameterError(self.__class__.__name__, item)

        path = self.kwargs['pdb_path']
        protein_size_limit = self.kwargs['protein_size_limit']

        try:
            # Check if PDB file exists locally, else download it
            if not os.path.exists(path + pdb + '.pdb'):
                pdbl = PDBList(verbose=False)
                pdbl.retrieve_pdb_file(
                    pdb, pdir=path, overwrite=False, file_format="pdb"
                )

                # Rename file to standard .pdb format from .ent
                os.rename(
                    path + "pdb" + pdb + ".ent", path + pdb + ".pdb"
                )
                # Confirm the file has been downloaded
                if not any(pdb in s for s in os.listdir(path)):
                    raise ValueError

            pdb_file = path + pdb + ".pdb"

            # Load the protein molecule from its PDB file
            m = Chem.MolFromPDBFile(pdb_file)
            n2 = m.GetNumAtoms()

            # Ensure protein is not too large for processing
            if n2 >= protein_size_limit:
                raise ProteinTooBig(n2, pdb_file)

            # Get adjacency matrix and positional data
            am = GetAdjacencyMatrix(m)
            c2 = m.GetConformers()[0]
            d2 = np.array(c2.GetPositions())

            # Extract binding pockets and construct corresponding DGL graphs
            pockets = _get_pockets_from_protein(pdb_file)
            constructed_graphs = _get_constructed_graphs_for_pockets(pockets, m, am, d2)

            # Batch graphs together for more efficient processing
            constructed_graphs = dgl.batch(constructed_graphs)
        except Exception as e:
            # Handle exceptions and return None
            # print(e)
            constructed_graphs = None

        return constructed_graphs

    def save_data(self, data: dgl.DGLGraph, path: str) -> None:
        dgl.save_graphs(path, [data])

    def load_data(self, path: str) -> dgl.DGLGraph:
        return dgl.load_graphs(path)[0][0]
