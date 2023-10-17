"""
dgl_graph_from_smiles.py

This module provides utilities and classes for preprocessing SMILES strings and converting them into DGLGraph objects.
It supports two modes of conversion:
1. Fragmentation of the SMILES string using the MacFrag tool, followed by graph construction for each fragment.
2. Direct conversion of the whole SMILES string into a graph representation.

Classes included:
- GraphFromPocketPreprocessor: Preprocessor class to convert SMILES strings into DGLGraph objects.

Utilities:
- _process_smile_graph: Helper function to process the SMILES string and construct molecular fragments as graphs using the MacFrag method.

Dependencies:
- dgl
- pandas
- dgllife
- rdkit
- local modules: macfrag, deepdrugdomain.exceptions, drug_preprocessor_factory, and data.data_preprocess

"""

from typing import Optional, List, Any

import dgl
import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from rdkit import Chem

from .macfrag import MacFrag
from deepdrugdomain.exceptions import MissingRequiredParameterError
from .drug_preprocessor_factory import DrugPreprocessorFactory
from ..base_preprocessor import BasePreprocessor
from deepdrugdomain.data.utils import serialize_dgl_graph_hdf5, deserialize_dgl_graph_hdf5


def _process_smile_graph(smile: str, max_block: int, max_sr: int, min_frag_atoms: int) -> Optional[List[dgl.DGLGraph]]:
    """
    Process the SMILES string to construct molecular fragments as graphs using the MacFrag method.

    Parameters:
    - smile: Input SMILES string.
    - max_block: Maximum number of blocks for the mac_frag function.
    - max_sr: Maximum SR value for the mac_frag function.
    - min_frag_atoms: Minimum number of atoms for a fragment.

    Returns:
    - List of DGLGraph objects or None.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        frags = MacFrag(mol, maxBlocks=max_block, maxSR=max_sr, asMols=False, minFragAtoms=min_frag_atoms)
    else:
        return None

    return frags


@DrugPreprocessorFactory.register("dgl_graph_from_smile_fragments")
class GraphFromSmilePreprocessor(BasePreprocessor):
    """
    Preprocessor class to convert SMILES strings to DGLGraph objects.

    This preprocessing class can operate in two modes:
    1. If the `fragment` option is true, the MacFrag tool from:
       https://github.com/yydiao1025/MacFrag/tree/main is used to fragment the SMILES, an approach inspired
       by the FragXsiteDTI paper. Each fragmented small molecule is then converted to a graph structure.
    2. If the `fragment` option is false, the whole SMILES molecule is directly converted into a graph structure.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, data: str) -> Optional[dgl.DGLGraph]:
        """
        Preprocess the input data series to construct a DGLGraph from SMILES string.

        Parameters:
        - data: Input data as pandas Series with a 'SMILE' key.
        - **kwargs: Additional parameters including max_block, max_sr, min_frag_atom, and fragment.

        Returns:
        - A batched DGLGraph or None.
        """
        smile = data
        if "fragment" not in self.kwargs:
            raise MissingRequiredParameterError(self.__class__.__name__, "fragment")

        fragment = self.kwargs['fragment']

        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

        if fragment:
            # Validate required parameters are provided
            required_keys = ["max_block", "max_sr", "min_frag_atom"]
            for item in required_keys:
                if item not in self.kwargs:
                    raise MissingRequiredParameterError(self.__class__.__name__, item)

            max_block = self.kwargs['max_block']
            max_sr = self.kwargs['max_sr']
            min_frag_atom = self.kwargs['min_frag_atom']

        try:
            if fragment:
                # Use the MacFrag method to fragment the SMILES and then construct a graph for each fragment.
                frags = _process_smile_graph(smile, max_block, max_sr, min_frag_atom)
                if frags is None:
                    return None
                smile_graphs = [smiles_to_bigraph(f, add_self_loop=True, node_featurizer=node_featurizer) for f in frags]
                constructed_graphs = dgl.batch(smile_graphs)
            else:
                # Construct a graph from the entire SMILES molecule.
                constructed_graphs = smiles_to_bigraph(smile, add_self_loop=True, node_featurizer=node_featurizer)

        except Exception as e:
            print(e)
            constructed_graphs = None

        return constructed_graphs

    # def _serialize_value(self, graph: dgl.DGLGraph) -> dict:
    #     """
    #     Serialize the DGL graph using the utility function.
    #     """
    #     return serialize_dgl_graph_hdf5(graph)
    #
    # def _deserialize_value(self, serialized_graph: dict) -> dgl.DGLGraph:
    #     """
    #     Deserialize the serialized DGL graph using the utility function.
    #     """
    #     return deserialize_dgl_graph_hdf5(serialized_graph)
    #
