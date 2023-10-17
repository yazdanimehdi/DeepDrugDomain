"""
simple_preprocessor.py

This module defines the SimpleProteinPreprocessor, which provides basic preprocessing
functions for proteins. It is registered to the DrugPreprocessFactory using a decorator.
"""
import os
from typing import Any, Optional

from Bio.PDB import PDBList

from deepdrugdomain.data.data_preprocess.base_preprocessor import BasePreprocessor
from .protein_preprocessing_factory import ProteinPreprocessorFactory



@ProteinPreprocessorFactory.register("simple_protein")
class SimpleProteinPreprocessor(BasePreprocessor):
    """
    A basic preprocessor for proteins.

    Provides fundamental preprocessing methods tailored for protein data.
    """

    def preprocess(self, data: str,
                   path: str):
        pdb = data.lower()
        try:
            if not os.path.exists(path + pdb + '.pdb'):
                pdbl = PDBList(verbose=False)
                pdbl.retrieve_pdb_file(
                    pdb, pdir=path, overwrite=False, file_format="pdb"
                )
                # Rename file to .pdb from .ent
                os.rename(
                    path + "pdb" + pdb + ".ent", path + pdb + ".pdb"
                )
                # Assert file has been downloaded
                assert any(pdb in s for s in os.listdir(path))

            constructed_graphs = process_protein(self.pdb_dir + pdb + ".pdb")
            self.p_graph[pdb] = constructed_graphs

        return data.upper()
