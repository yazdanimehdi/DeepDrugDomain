"""
The FingerprintFromSequencePreprocessor class is designed to convert biological sequences (e.g., proteins, DNA) into numerical representations or 'fingerprints' using various bioinformatics methods. These fingerprints can then be used for further analysis or as input to machine learning models in computational biology and bioinformatics.

This class supports several sequence processing methods, each of which translates a sequence into a different type of numerical representation based on the properties of amino acids or nucleotides in the sequence.

Supported Methods:
- 'quasi': Calculates the Quasi-Sequence-Order descriptor, a method that captures the global sequence order information.
- 'aac': Computes the Amino Acid Composition descriptor, representing the fraction of each type of amino acid in a protein sequence.
- 'paac': Calculates the Pseudo-Amino Acid Composition, which combines the conventional amino acid composition with additional sequence-order information.
- 'ct': Uses the Conjoint Triad method, considering the properties of one amino acid and its neighboring amino acids.
- 'custom': Allows for a custom function to be specified for sequence processing, providing flexibility for bespoke analysis.

Each method has unique applications in sequence analysis, enabling this class to be versatile in handling various types of sequence data.

Parameters:
- method (str, default 'quasi'): The method to use for sequence processing.
- **kwargs: Additional keyword arguments.

The main functionality is implemented in the `preprocess` method, which takes a biological sequence as input and returns a numerical representation as a torch tensor. The method selected determines the specific type of fingerprint generated.

Raises:
- ValueError: If an invalid method is specified.
"""


from typing import Optional
import torch
from deepdrugdomain.utils.exceptions import MissingRequiredParameterError
from ..factory import PreprocessorFactory
from ..base_preprocessor import BasePreprocessor
from ..utils import GetQuasiSequenceOrder, CalculateAADipeptideComposition, GetPseudoAAC, CalculateConjointTriad
import numpy as np


@PreprocessorFactory.register("sequence_to_fingerprint", "protein_sequence", "fingerprint")
class FingerprintFromSequencePreprocessor(BasePreprocessor):
    def __init__(self, method: str = 'quasi',
                 **kwargs):
        """
        Initialize the FingerprintFromSequencePreprocessor with the specified method.

        This constructor sets up the preprocessor for generating numerical representations of biological sequences using the specified method. It extends the BasePreprocessor class and allows for additional keyword arguments to be passed and handled by the base class.

        Parameters:
        - method (str, default 'quasi'): The method to be used for converting sequences to fingerprints. Supported methods are 'quasi', 'aac', 'paac', 'ct', and 'custom'.
        - **kwargs: Additional keyword arguments to be passed to the BasePreprocessor class.

        Supported Methods:
        - 'quasi': Quasi-Sequence-Order descriptor.
        - 'aac': Amino Acid Composition descriptor.
        - 'paac': Pseudo-Amino Acid Composition.
        - 'ct': Conjoint Triad method.
        - 'custom': Custom function specified by the user for sequence processing.
        """
        super().__init__(**kwargs)
        self.method = method

    def preprocess(self, sequence: str) -> Optional[torch.Tensor]:
        """
        Generate a numerical representation (fingerprint) of a biological sequence.

        This method processes a given biological sequence (e.g., a protein or DNA sequence) and converts it into a numerical fingerprint based on the specified method. The output is a torch tensor representing this fingerprint, which can be used for further analysis or as input to machine learning models.

        Parameters:
        - sequence (str): The biological sequence to be processed.

        Returns:
        - Optional[torch.Tensor]: A torch tensor representing the sequence's fingerprint. Returns None if an error occurs or if the sequence cannot be processed.

        The method utilizes different sequence processing techniques based on the 'method' attribute set during initialization. It handles exceptions during processing and returns None in case of an error.

        Raises:
        - ValueError: If an invalid method is specified or if the sequence cannot be processed using the selected method.
        """

        valid_methods = ['quasi', 'aac', 'paac', 'ct', 'custom']
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method specified. Choose from {valid_methods}.")

        fingerprints = None

        try:
            if self.method == 'quasi':
                features = GetQuasiSequenceOrder(sequence)
                features = np.array(features)

            elif self.method == 'aac':
                features = CalculateAADipeptideComposition(sequence)
                features = np.array(features)

            elif self.method == 'paac':
                features = GetPseudoAAC(sequence)
                features = np.array(features)

            elif self.method == 'ct':
                features = CalculateConjointTriad(sequence)
                features = np.array(features)

            elif self.method == 'custom':
                fingerprints = self.custom_fingerprint(sequence)

        except Exception as e:
            print(
                f'Error processing Sequence {sequence} with method {self.method}: {e}')

        return torch.tensor(fingerprints, dtype=torch.float) if fingerprints is not None else None
