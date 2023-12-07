"""
This module provides functionalities for extracting protein motifs from sequences 
using the Genome.jp website's motif search tool and sequence alignment techniques 
based on the PROSITE database. The motif extraction is achieved by submitting 
protein sequences to an online search form and parsing the returned motifs.

The PROSITE database, referenced in this module, is a resource for the 
identification of protein families and domains. It provides tools for the alignment 
of sequences and the detection of conserved patterns, aiding in the identification 
of functionally important sites.

References:
- Falquet L., Pagni M., Bucher P., Hulo N., Sigrist C.J, Hofmann K., Bairoch A. 
  "The PROSITE database, its status in 2002" Nucl. Acids Res. 30(1):235-238, 2002.
  PubMed: 11752303
- Gribskov, M., McLachlan, A.D., Eisenberg, D. "Profile analysis: detection of 
  distantly related proteins." Proc. Natl. Acad. Sci. USA 84:4355-4358 (1987).
  PubMed: 87260806
- Gribskov M., Luethy, R., Eisenberg, D. "Profile analysis." Methods Enzymol. 
  183: 146-159 (1990). PubMed: 90190364
- Bucher P., Bairoch A. "A generalized profile syntax for biomolecular sequences 
  motifs and its function in automatic sequence interpretation." In "ISMB-94; 
  Proceedings 2nd International Conference on Intelligent Systems for Molecular 
  Biology." (Altman R., Brutlag D., Karp P., Lathrop R., Searls D., Eds.), 
  pp53-61, AAAIPress, Menlo Park, (1994). PubMed: 96039003
- Luethy R., Xenarios I., Bucher P. "Improving the sensitivity of the sequence 
  profile method." Prot. Sci. 3:139-146 (1994). PubMed: 7511453
- Bucher P., Karplus K., Moeri N., Hofmann, K. "A flexible motif search technique 
  based on generalized profiles." Comput. Chem. 20:3-24 (1996).

This module includes functions for requesting and parsing motif data and a 
preprocessor class for handling the conversion of protein sequences to motif 
representations, leveraging the described databases and methodologies.

Dependencies:
- requests: For making HTTP requests to the Genome.jp website.
- BeautifulSoup (bs4): For parsing the HTML responses.
- NumPy and Torch: For handling the array and tensor operations.

Note: The effectiveness of motif extraction and subsequent analysis depends 
significantly on the quality of the input sequence and the appropriateness 
of the motifs extracted from the Genome.jp tool and PROSITE database.
"""

from collections import defaultdict
import os
from typing import Any, Optional
from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
import numpy as np
import torch

import requests
import re
from bs4 import BeautifulSoup


def get_motifs(sequence):
    """
    Retrieves the first motif from a given protein sequence by querying an online motif search tool.

    Args:
        sequence (str): The protein sequence for which to find the motif.

    Returns:
        Optional[str]: The first motif found for the given sequence. If no motif is found, returns None.

    Note:
        The function queries the 'https://www.genome.jp/tools-bin/search_motif_lib' URL,
        sending the protein sequence and specific form data to search for motifs in the PROSITE format.
    """

    url = "https://www.genome.jp/tools-bin/search_motif_lib"
    form_data = {
        "seq": sequence,
        "FORMAT": "PROSITE",
        "prosite_pattern": "on",
        "pfam": "on",
        # "prosite_matrix": "on",
        "skip_entry": "on",
        "skip_unspecific_profile": "on"
    }
    response = requests.post(url, data=form_data)
    soup = BeautifulSoup(response.text, 'html.parser')
    motifs = soup.find_all('input', {'type': 'hidden', 'name': 'FOUND'})
    if not motifs:
        motifs = soup.find_all('input', {'type': 'hidden', 'name': 'FNDSEQ'})
    if motifs:
        motifs = motifs[0].get('value').split(',')[0]
        return motifs

    return None


def ps(x, word_len):
    """
    Generates tuples of substrings (words) from the input string, each of a specified length.

    Args:
        x (str): The input string from which to generate the substrings.
        word_len (int): The length of each substring (word).

    Returns:
        tuple: A tuple containing all the substrings (words) of the specified length extracted from the input string.
    """
    t = ()
    for i in range(word_len):
        y = len(x)
        for m in range(i, y, word_len):
            k = x[m:m+word_len]
            if (len(k) == word_len):
                t = t+(k,)
    return t


@PreprocessorFactory.register("sequence_to_motif", "protein_sequence", "motif_tensor")
class ProteinMotifPreprocessor(BasePreprocessor):
    """
    A preprocessor class for converting protein sequences to motifs and representing them in a specified format.

    This class is capable of handling one-hot encoding and fixed-length representation of the motifs.

    Attributes:
        ngram (int): The length of each word (substring) to consider in the motif.
        word_dict (Optional[dict]): A dictionary mapping words to indices, used for one-hot encoding.
        max_length (Optional[int]): The maximum length of the motif representation.
        one_hot (bool): Flag indicating whether to use one-hot encoding.
        number_of_combinations (Optional[int]): The number of possible combinations for one-hot encoding.

    Methods:
        preprocess(data: str) -> torch.Tensor:
            Converts a given protein sequence into a tensor representation of its motif.
            The representation depends on the one_hot flag and other attributes of the class.
    """

    def __init__(self, ngram: int, word_dict: Optional[dict] = None, max_length: Optional[int] = None, one_hot: bool = True, number_of_combinations: Optional[int] = None, **kwargs):
        """
        Initializes the ProteinMotifPreprocessor with given parameters.

        Args:
            ngram (int): The length of each word (substring) in the motif.
            word_dict (Optional[dict]): A dictionary mapping words to indices for one-hot encoding.
            max_length (Optional[int]): The maximum length of the motif representation.
            one_hot (bool): If True, uses one-hot encoding for the motifs.
            number_of_combinations (Optional[int]): The total number of combinations for one-hot encoding.

        Raises:
            AssertionError: If one_hot is True, asserts that both number_of_combinations and max_length are provided.
        """

        super().__init__(**kwargs)
        self.word_dict = word_dict
        self.ngram = ngram
        self.max_length = max_length
        self.one_hot = one_hot
        self.number_of_combinations = number_of_combinations
        if one_hot:
            assert number_of_combinations is not None and max_length is not None, "If one_hot is True, number_of_combinations and max_length must be specified."

    def preprocess(self, data: str) -> torch.Tensor:
        """
        Converts a protein sequence into a motif representation as a torch.Tensor.

        The representation depends on the one_hot attribute and other class parameters.

        Args:
            data (str): The protein sequence to preprocess.

        Returns:
            torch.Tensor: A tensor representation of the protein's motif. Returns None if no motif is found or
                          if the motif length exceeds max_length when one_hot is True.
        """
        motif = get_motifs(data)

        if motif is None:
            print("No motif found for sequence: ", data)
            return None

        self.word_dict = self.word_dict if self.word_dict is not None else defaultdict(
            lambda: len(self.word_dict))
        words = ps(motif, self.ngram)
        if self.one_hot:
            if len(words) > self.max_length:
                return None
            one_hot = np.zeros((self.max_length, self.number_of_combinations))
            for i, word in enumerate(words):
                one_hot[i, self.word_dict[word]] = 1
            return torch.from_numpy(one_hot)
        else:
            words = np.array([self.word_dict[word] for word in words])
            if self.max_length - len(words) > 0:
                words = np.pad(words, (0, self.max_length - len(words)),
                               'constant', constant_values=0)
                return torch.tensor(words, dtype=torch.long)

            else:
                return None
