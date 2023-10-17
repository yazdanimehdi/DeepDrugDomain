from typing import Optional, List
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
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
        frags = MacFrag(mol, maxBlocks=max_block, maxSR=max_sr, asMols=False, minFragAtoms=min_frag_atoms)
    else:
        return None

    return frags


@PreprocessorFactory.register("dgl_graph_from_smile")
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
                # Use the MacFrag method to fragment the SMILES and then construct a graph for each fragment.
                frags = _process_smile_graph(smile, max_block, max_sr, min_frag_atom)
                if frags is None:
                    return None
                smile_graphs = [smiles_to_bigraph(f, add_self_loop=True, node_featurizer=node_featurizer) for f in frags]
                constructed_graphs = dgl.batch(smile_graphs)

            except Exception as e:
                constructed_graphs = None

        else:
            try:
                # Construct a graph from the entire SMILES molecule.
                constructed_graphs = smiles_to_bigraph(smile, add_self_loop=True, node_featurizer=node_featurizer)

            except Exception as e:
                print(e)
                constructed_graphs = None

        return constructed_graphs

    def save_data(self, data: dgl.DGLGraph, path: str) -> None:
        dgl.save_graphs(path, [data])

    def load_data(self, path: str) -> dgl.DGLGraph:
        return dgl.load_graphs(path)[0][0]