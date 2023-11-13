from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from deepdrugdomain.data.preprocessing.factory import PreprocessorFactory
from typing import Optional, Callable
import torch
from .train_model import seq_to_kmers, train_or_load_word2vec_model
import numpy as np
import os


@PreprocessorFactory.register("word2vec")
class Word2VecPreprocessor(BasePreprocessor):
    def __init__(self, model_path: str, vec_size: int, sentences_path: Optional[str] = None, sentence_preprocessing: Optional[Callable] = seq_to_kmers, **kwargs):
        """
        A preprocessor class for generating Word2Vec embeddings for protein sequences.

        This class is capable of either loading a pre-trained Word2Vec model or training a new one
        if the model path does not exist. It uses an optional custom sentence preprocessing function.

        Attributes:
        - model_path (str): Path to the trained Word2Vec model.
        - vec_size (int): The size of the word vectors.
        - sentences_path (str, optional): Path to the sentences to be used for training the model.
        - sentence_preprocessing (Callable, optional): Function to preprocess sentences into the
          format expected by the Word2Vec model (default is seq_to_kmers).

        Raises:
        - AssertionError: If the `model_path` does not exist and `sentences_path` is None.

        Inherited Attributes:
        - **kwargs: Additional keyword arguments passed to the BasePreprocessor.
        """
        super().__init__(**kwargs)
        if os.path.exists(model_path):
            self.model = train_or_load_word2vec_model(
                load_model_path=model_path)
        else:
            assert sentences_path is not None, "sentences_path must be provided if model_path does not exist"
            self.model = train_or_load_word2vec_model(
                sentences_path=sentences_path, save_model_path=model_path, vector_size=vec_size, **kwargs)

        self.sentence_preprocessing = sentence_preprocessing
        self.vec_size = vec_size

    def preprocess(self, data: str) -> torch.Tensor:
        """
        Preprocess a protein sequence to generate its Word2Vec embedding as a torch tensor.

        The method applies the sentence preprocessing function to the input data to obtain a list
        of words, then generates a corresponding vector embedding for each word. The result is
        a tensor of shape (len(data), vec_size).

        Parameters:
        - data (str): A string representing the protein sequence.

        Returns:
        - A torch.Tensor representing the sequence's Word2Vec embeddings.
        """
        data = self.sentence_preprocessing(data)
        vec = np.zeros((len(data), self.vec_size))
        i = 0
        for word in data:
            vec[i, ] = self.model.wv[word]
            i += 1

        return torch.from_numpy(vec)
