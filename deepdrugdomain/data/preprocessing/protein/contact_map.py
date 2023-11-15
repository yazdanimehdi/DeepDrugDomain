"""
This module provides functionality to generate contact maps from protein structures in PDB format. Contact maps are 
2D representations of protein structures where each element indicates the proximity of two amino acids in the 3D space.

Classes:
    ContactMapFromPDBPreprocessor: Generates a contact map from a PDB file.

Methods:
    __init__: Initializes the preprocessor with the specified parameters and PDB file path.
    preprocess: Generates the contact map from the PDB file based on the initialized PDB path.
"""


from typing import Optional
import torch
import numpy as np
from Bio.PDB import PDBParser
from ..base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
from ..utils import download_pdb


@PreprocessorFactory.register('contact_map_from_pdb')
class ContactMapFromPDBPreprocessor(BasePreprocessor):
    def __init__(self, pdb_path: str, method: str = 'c_alpha', distance_threshold: float = 3.8, normalize_distance: bool = True, **kwargs):
        """
        Initializes the ContactMapFromPDBPreprocessor with the specified PDB file path and parameters for contact map calculation.

        Parameters:
            pdb_path (str): Path to the PDB file or a PDB ID to be downloaded.
            method (str, default 'c_alpha'): Method used for determining the atoms used in distance calculations. 
                                             Supported methods are:
                                             - 'c_alpha': Uses only the C-alpha atoms of the residues.
                                             - 'all_atoms': Uses all atoms of the residues.
                                             - 'custom_method': A custom method to be implemented by the user.
            distance_threshold (float, default 3.8): The threshold distance for considering two amino acids to be in contact.
            normalize_distance (bool, default True): Whether to normalize the distances using the given threshold.
            **kwargs: Additional keyword arguments to be passed to the BasePreprocessor class.
        """
        super().__init__(**kwargs)
        self.method = method
        self.distance_threshold = distance_threshold
        self.normalize_distance = normalize_distance
        self.pdb_path = pdb_path

    def preprocess(self, pdb: str) -> Optional[torch.Tensor]:
        """
        Generates the contact map from the PDB file.

        Parameters:
            pdb (str): PDB ID to find in the pdb_path or to be downloaded.

        Returns:
            Optional[torch.Tensor]: A tensor representing the contact map of the protein structure. The tensor is of shape
                                    (num_atoms, num_atoms) with each element indicating whether the corresponding amino acids
                                    are within the specified distance threshold. Returns None if an error occurs.

        Raises:
            ValueError: If an invalid method is specified.
            Exception: If there is an error in processing the PDB file.
        """
        try:
            parser = PDBParser()
            pdb_path = download_pdb(pdb, self.pdb_path)
            structure = parser.get_structure('protein', pdb_path)
            model = structure[0]  # Assuming the PDB has only one model

            if self.method == 'c_alpha':
                atoms = [res['CA']
                         for res in model.get_residues() if 'CA' in res]
            elif self.method == 'all_atoms':
                atoms = [atom for res in model.get_residues()
                         for atom in res if res.get_id()[0] == ' ']
            elif self.method == 'custom_method':
                # Implement custom method here
                atoms = []
            else:
                raise ValueError(
                    f"Invalid method: {self.method}. Supported methods are 'c_alpha', 'all_atoms', and 'custom_method'.")

            num_atoms = len(atoms)
            contact_map = np.zeros((num_atoms, num_atoms), dtype=np.float32)

            for i, atom1 in enumerate(atoms):
                # To avoid duplicate calculations
                for j, atom2 in enumerate(atoms[i+1:], i+1):
                    distance = (atom1 - atom2).get_vector().norm()
                    if self.normalize_distance:
                        contact_map[i, j] = contact_map[j, i] = 1 / \
                            (1 + distance / self.distance_threshold)
                    else:
                        contact_map[i, j] = contact_map[j,
                                                        i] = 1 if distance <= self.distance_threshold else 0

            return torch.from_numpy(contact_map)

        except Exception as e:
            print(f'Error processing PDB file {pdb_path}: {e}')
            return None
