"""
The `FingerprintFromSmilePreprocessor` class provides functionality for generating various types of molecular fingerprints from a SMILES string. This class supports multiple fingerprinting methods, each suitable for different types of molecular analysis and modeling in cheminformatics.

Supported Fingerprinting Methods:
- 'rdkit': RDKit's topological fingerprint, a path-based fingerprint accounting for molecular topology.
  Reference: RDKit: Open-source cheminformatics; http://www.rdkit.org
- 'morgan': Morgan fingerprints (Circular fingerprints), equivalent to ECFP (Extended-Connectivity Fingerprints).
  Reference: Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5), 742-754.
- 'daylight': Daylight-like topological fingerprints, capturing molecular connectivity.
  Reference: Daylight Theory Manual; https://www.daylight.com/dayhtml/doc/theory/
- 'ErG': Reduced graph fingerprint by RDKit, which provides a high-level abstracted representation of molecules.
  Reference: RDKit: Open-source cheminformatics; http://www.rdkit.org
- 'rdkit2d': A set of RDKit 2D normalized descriptors, providing a comprehensive description of 2D molecular structure.
  Reference: Landrum, G. (2006). RDKit: Open-source cheminformatics.
- 'pubchem': PubChem fingerprints, designed for the PubChem chemical structure database.
  Reference: Bolton, E. E., Wang, Y., Thiessen, P. A., & Bryant, S. H. (2008). PubChem: integrated platform of small molecules and biological activities. Annual reports in computational chemistry, 4, 217-241.
- 'ammvf': Custom fingerprint method, allowing for tailored feature extraction based on atom, bond, and edge dictionaries.
- 'custom': User-defined custom fingerprint function, providing flexibility for bespoke fingerprinting algorithms.

Each method has its own characteristics and is chosen based on the specific requirements of the analysis or model being developed.

Parameters:
- method (str): The method to use for fingerprint generation.
- radius (Optional[int]): The radius parameter for Morgan fingerprints.
- nBits (Optional[int]): The size of the fingerprint bit vector for Morgan fingerprints.
- num_finger (Optional[int]): The number of bits for the Daylight-type fingerprint.
- atom_dict, bond_dict, fingerprint_dict, edge_dict (Optional[Dicts]): Custom dictionaries for the 'ammvf' method.
- consider_hydrogen (bool): Whether to consider hydrogen atoms in the fingerprint.
- custom_fingerprint (Optional[Callable]): A custom function for fingerprint generation.

The class inherits from `BasePreprocessor`, and its main functionality is implemented in the `preprocess` method, which takes a SMILES string and returns a fingerprint as a torch tensor.

Example usage:
>>> preprocessor = FingerprintFromSmilePreprocessor(method='morgan')
>>> smile = 'CCO'
>>> fingerprint = preprocessor.preprocess(smile)
"""


from collections import defaultdict
from typing import Dict, Optional, List, Callable, Union
from rdkit import Chem, DataStructs
import torch
from ..utils import calcPubChemFingerAll, rdNormalizedDescriptors, create_atoms, create_ij_bond_dict, extract_fingerprints
from deepdrugdomain.utils.exceptions import MissingRequiredParameterError
from ..factory import PreprocessorFactory
from ..base_preprocessor import BasePreprocessor
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

AtomDictType = Dict[Union[str, tuple], int]
BondDictType = Dict[str, int]
FingerprintDictType = Dict[Union[int, tuple], int]


@PreprocessorFactory.register("smile_to_fingerprint")
class FingerprintFromSmilePreprocessor(BasePreprocessor):
    def __init__(self, method: str = 'rdkit',
                 radius: Optional[int] = 2,
                 nBits: Optional[int] = 1024,
                 num_finger: Optional[int] = 2048,
                 atom_dict: Optional[AtomDictType] = None,
                 bond_dict: Optional[BondDictType] = None,
                 fingerprint_dict: Optional[FingerprintDictType] = None,
                 edge_dict: Optional[Dict] = None,
                 consider_hydrogen: bool = False,
                 custom_fingerprint: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize the FingerprintFromSmilePreprocessor with specified parameters.

        Parameters:
        - method (str, default 'rdkit'): The fingerprinting method to be used. Supported methods include 'rdkit', 
        'morgan', 'daylight', 'ErG', 'rdkit2d', 'pubchem', 'ammvf', and 'custom'.
        - radius (Optional[int], default 2): The radius for Morgan fingerprint calculation.
        - nBits (Optional[int], default 1024): The size of the bit vector for Morgan fingerprints.
        - num_finger (Optional[int], default 2048): The number of bits for Daylight-type fingerprints.
        - atom_dict (Optional[AtomDictType]): Custom dictionary mapping atom representations to integers for 'ammvf' method.
        - bond_dict (Optional[BondDictType]): Custom dictionary mapping bond types to integers for 'ammvf' method.
        - fingerprint_dict (Optional[FingerprintDictType]): Custom dictionary for mapping fingerprints to indices for 'ammvf' method.
        - edge_dict (Optional[Dict]): Custom dictionary for edge representation for 'ammvf' method.
        - consider_hydrogen (bool, default False): Flag to determine if hydrogen atoms should be included in the molecule representation.
        - custom_fingerprint (Optional[Callable]): A custom function for fingerprint generation for 'custom' method.

        This initializer sets up the preprocessor with the specified method and parameters, enabling various types of 
        molecular fingerprint generation from SMILES strings. The method chosen determines the type of fingerprint and 
        the applicable parameters.

        Raises:
        - MissingRequiredParameterError: If a required parameter for a specific method is not provided.
        """

        super().__init__(**kwargs)
        self.consider_hydrogen = consider_hydrogen
        self.method = method
        self.radius = radius
        self.nBits = nBits
        self.num_finger = num_finger
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.fingerprint_dict = fingerprint_dict
        self.edge_dict = edge_dict
        self.custom_fingerprint = custom_fingerprint

    def preprocess(self, smiles: str) -> Optional[torch.Tensor]:
        """
        Generate a molecular fingerprint based on a SMILES string using various methods.

        Parameters:
        - smiles (str): SMILES string of the molecule.

        Returns:
        - Optional[torch.Tensor]: A tensor representing the molecular fingerprint. None if an error occurs.

        Supported methods:
        - 'rdkit': RDKit fingerprint
        - 'morgan': Morgan fingerprint
        - 'daylight': Daylight-type fingerprint
        - 'ErG': ErG fingerprint
        - 'rdkit2d': RDKit 2D normalized descriptors
        - 'pubchem': PubChem fingerprint
        - 'ammvf': Custom fingerprint
        - 'custom': User-defined custom fingerprint function
        """
        valid_methods = ['rdkit', 'morgan', 'daylight',
                         'ErG', 'rdkit2d', 'pubchem', 'ammvf', 'custom']
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method specified. Choose from {valid_methods}.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.consider_hydrogen:
            mol = Chem.AddHs(mol)

        fingerprints = None

        try:
            if self.method == 'rdkit':
                fingerprints = Chem.RDKFingerprint(mol)

            elif self.method == 'morgan':
                features_vec = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.nBits)
                fingerprints = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(features_vec, fingerprints)

            elif self.method == 'daylight':
                bv = FingerprintMols.FingerprintMol(mol)
                temp = tuple(bv.GetOnBits())
                fingerprints = np.zeros((self.num_finger, ))
                fingerprints[np.array(temp)] = 1

            elif self.method == 'ErG':
                fingerprints = np.array(GetErGFingerprint(mol))

            elif self.method == 'rdkit2d':
                generator = rdNormalizedDescriptors.RDKit2DNormalized()
                features = np.array(generator.process(smiles)[1:])
                features[np.isnan(features)] = 0
                fingerprints = features

            elif self.method == 'pubchem':
                fingerprints = calcPubChemFingerAll(smiles)

            elif self.method == 'ammvf':
                self.atom_dict = self.atom_dict if self.atom_dict is not None else defaultdict(
                    lambda: len(self.atom_dict))
                self.bond_dict = self.bond_dict if self.bond_dict is not None else defaultdict(
                    lambda: len(self.bond_dict))
                self.fingerprint_dict = self.fingerprint_dict if self.fingerprint_dict is not None else defaultdict(
                    lambda: len(self.fingerprint_dict))
                self.edge_dict = self.edge_dict if self.edge_dict is not None else defaultdict(
                    lambda: len(self.edge_dict))
                atoms = create_atoms(mol, self.atom_dict)
                ij_bond_dict = create_ij_bond_dict(mol, self.bond_dict)
                fingerprints = extract_fingerprints(
                    atoms, ij_bond_dict, self.radius, self.fingerprint_dict, self.edge_dict)

            elif self.method == 'custom':
                fingerprints = self.custom_fingerprint(mol)

        except Exception as e:
            print(
                f'Error processing SMILES {smiles} with method {self.method}: {e}')

        return torch.tensor(fingerprints, dtype=torch.float) if fingerprints is not None else None
