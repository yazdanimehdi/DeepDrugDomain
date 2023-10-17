from typing import Optional, List, Union

import dgl
import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from rdkit import Chem

from .macfrag import mac_frag
from deepdrugdomain.exceptions import MissingRequiredParameterError
from .drug_preprocessor_factory import DrugPreprocessorFactory
from deepdrugdomain.data.data_preprocess import BasePreprocessor


def _process_smile_graph(smile: str, max_block: int, max_sr: int, min_frag_atoms: int) -> Optional[List[dgl.DGLGraph]]:
    """
    Process the SMILES string to construct molecular fragments as graphs.

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
        frags = mac_frag(mol, max_blocks=max_block, max_sr=max_sr, as_mols=False, min_frag_atoms=min_frag_atoms)
    else:
        return None

    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    g = [dgl.add_self_loop(smiles_to_bigraph(fr, node_featurizer=node_featurizer)) for fr in frags]

    return g


@DrugPreprocessorFactory.register("dgl_graph_from_smile_fragments")
class GraphFromPocketPreprocessor(BasePreprocessor):
    """
    Preprocessor class to convert SMILES strings to DGLGraph objects.

    This preprocessing class utilizes the MacFrag tool from:
    https://github.com/yydiao1025/MacFrag/tree/main to fragment small molecules.
    Each fragment is then converted to a graph structure.

    This approach is inspired by the FragXsiteDTI paper.

    """

    def preprocess(self, data: pd.Series, **kwargs) -> Optional[dgl.DGLGraph]:
        """
        Preprocess the input data series to construct a DGLGraph from SMILES string.

        Parameters:
        - data: Input data as pandas Series with a 'SMILE' key.
        - **kwargs: Additional parameters including max_block, max_sr, and min_frag_atom.

        Returns:
        - A batched DGLGraph or None.
        """

        if 'SMILE' not in data:
            raise ValueError(
                "The input data series does not contain the 'SMILE' key. Please ensure the series is correctly formatted.")

        smile = data['SMILE']

        # Validate required parameters are provided
        required_keys = ["max_block", "max_sr", "min_frag_atom"]
        for item in required_keys:
            if item not in kwargs:
                raise MissingRequiredParameterError(self.__class__.__name__, item)

        max_block = kwargs['max_block']
        max_sr = kwargs['max_sr']
        min_frag_atom = kwargs['min_frag_atom']

        try:
            smile_graphs = _process_smile_graph(smile, max_block, max_sr, min_frag_atom)
            constructed_graphs = dgl.batch(smile_graphs)

        except Exception as e:
            print(e)
            constructed_graphs = None

        return constructed_graphs
