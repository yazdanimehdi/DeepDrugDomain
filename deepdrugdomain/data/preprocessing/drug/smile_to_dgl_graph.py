from functools import partial
from typing import Optional, List, Callable
import dgl
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from rdkit import Chem
from .macfrag import MacFrag
from deepdrugdomain.utils.exceptions import MissingRequiredParameterError
from ..factory import PreprocessorFactory
from ..base_preprocessor import BasePreprocessor


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
        frags = MacFrag(mol, maxBlocks=max_block, maxSR=max_sr,
                        asMols=False, minFragAtoms=min_frag_atoms)
    else:
        return None

    return frags


@PreprocessorFactory.register("smile_to_dgl_graph", "smile", "graph")
class GraphFromSmilePreprocessor(BasePreprocessor):
    """
    Preprocessor class to convert SMILES strings to DGLGraph objects.

    This preprocessing class can operate in two modes:
    1. If the `fragment` option is true, the MacFrag tool from:
       https://github.com/yydiao1025/MacFrag/tree/main is used to fragment the SMILES, an approach inspired
       by the FragXsiteDTI paper. Each fragmented small molecule is then converted to a graph structure.
    2. If the `fragment` option is false, the whole SMILES molecule is directly converted into a graph structure.

    """

    def __init__(self, node_featurizer: Callable, edge_featurizer: Optional[Callable] = None, consider_hydrogen: bool = False, fragment: bool = False, hops: int = 1,  **kwargs):
        super().__init__(**kwargs)
        self.consider_hydrogen = consider_hydrogen
        self.fragment = fragment
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.hops = hops

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

        if self.fragment:
            # Validate required parameters are provided
            required_keys = ["max_block", "max_sr", "min_frag_atom"]
            for item in required_keys:
                if item not in self.kwargs:
                    raise MissingRequiredParameterError(
                        self.__class__.__name__, item)

            max_block = self.kwargs['max_block']
            max_sr = self.kwargs['max_sr']
            min_frag_atom = self.kwargs['min_frag_atom']
            try:
                # Use the MacFrag method to fragment the SMILES and then construct a graph for each fragment.
                frags = _process_smile_graph(
                    smile, max_block, max_sr, min_frag_atom)
                if frags is None:
                    return None
                smile_graphs = [smiles_to_bigraph(
                    f, add_self_loop=True, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for f in frags]
                constructed_graphs = smile_graphs

            except Exception as e:
                constructed_graphs = None

        else:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(
                    smile)) if self.consider_hydrogen else Chem.MolFromSmiles(smile)
                # Construct a graph from the entire SMILES molecule.
                constructed_graphs = mol_to_bigraph(
                    mol, add_self_loop=True, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer)

                if self.hops > 1:
                    constructed_graphs = dgl.khop_graph(
                        constructed_graphs, self.hops)

            except Exception as e:
                print(e)
                constructed_graphs = None

        return constructed_graphs

    def save_data(self, data: dgl.DGLGraph, path: str) -> None:
        if not isinstance(data, dgl.DGLGraph):
            super().save_data(data, path)
        else:
            dgl.save_graphs(path, [data])

    def load_data(self, path: str) -> dgl.DGLGraph:
        if self.fragment:
            return super().load_data(path)
        else:
            return dgl.load_graphs(path)[0][0]
