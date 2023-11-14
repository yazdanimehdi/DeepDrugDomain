"""
This module provides utilities for converting sequences into k-mers and 
training Word2Vec models using the Gensim library. It includes a function 
to create k-mers from sequences and a class to represent a corpus of sequences 
in dataframe format, suitable for training Word2Vec models. It also contains 
a generic function to train or load Word2Vec models from a given corpus or 
pretrained model, with the ability to save and update the vocabulary and 
train new sentences.

Examples:
>>> seq = 'ATCGATCGA'
>>> kmers = seq_to_kmers(seq, k=3)
>>> print(kmers)
['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG', 'CGA']

>>> data = ['ATCGATCGA', 'CGATCGATC']
>>> corpus = Corpus(data, ngram=3)
>>> for kmer_sentence in corpus:
...     print(kmer_sentence)
['ATC', 'TCG', 'CGA', 'GAT', 'ATC', 'TCG', 'CGA']
['CGA', 'GAT', 'ATC', 'TCG', 'CGA', 'GAT', 'ATC']

>>> # Assuming 'sentences' is an Iterable of tokenized sentences:
>>> model_path = 'path/to/model'
>>> model = train_or_load_word2vec_model(sentences=sentences, save_model_path=model_path)
>>> loaded_model = train_or_load_word2vec_model(load_model_path=model_path)
"""


from gensim.models import Word2Vec
import gensim.downloader as api
from typing import Iterable, List, Optional
import os
import pandas as pd


def seq_to_kmers(seq: str,
                 k: int = 3):
    """ Divide a string into a list of kmers strings.

    Parameters:
        seq (string)
        k (int), default 3
    Returns:
        List containing a list of kmers.
    """
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]


class Corpus:
    """ An iterable object for training word2vec models. """

    def __init__(self, df: List[str], ngram: int = 3):
        self.df = df
        self.ngram = ngram

    def __iter__(self):
        for sentence in self.df:
            yield seq_to_kmers(sentence, self.ngram)


def train_or_load_word2vec_model(sentences: List[str] = None,
                                 ngram: Optional[int] = 3,
                                 vector_size: int = 100,
                                 window: int = 5,
                                 min_count: int = 1,
                                 workers: int = 4,
                                 epochs: int = 30,
                                 save_model_path: Optional[str] = None,
                                 load_model_path: Optional[str] = None,
                                 update_vocab: bool = False) -> Word2Vec:
    """
    Train a new Word2Vec model or continue training an existing one, with options to
    save or load the model to/from disk, and to use a chosen pretrained model.

    :param sentences: List of sentences without extracting the words (default: None)
    :param vector_size: Dimensionality of the word vectors (default: 100)
    :param window: Maximum distance between current and predicted word within a sentence (default: 5)
    :param min_count: Ignores all words with total frequency lower than this (default: 1)
    :param workers: Number of worker threads to train the model (default: 4)
    :param epochs: Number of training iterations over the corpus (default: 5)
    :param save_model_path: Path to save the trained model (default: None)
    :param load_model_path: Path to load a saved Word2Vec model instead of training a new one (default: None)
    :param update_vocab: Whether to update the model's vocabulary with new sentences (default: False)
    :return: Word2Vec model
    """

    if load_model_path and os.path.exists(load_model_path):
        model = Word2Vec.load(load_model_path)

    else:
        model = Word2Vec(vector_size=vector_size, window=window,
                         min_count=min_count, workers=workers)

    # Build vocabulary and train model
    if sentences is not None:
        sentences = [seq_to_kmers(seq, ngram) for seq in sentences]
        if update_vocab and load_model_path:
            # Update the existing vocabulary
            model.build_vocab(sentences, update=True)
        else:
            model.build_vocab(sentences)  # Build vocabulary from scratch

            model.train(sentences, total_examples=len(sentences),
                        epochs=epochs)  # Train the model

    # Save the model if a save path is provided
    if save_model_path:
        model.save(save_model_path)

    print('Word2Vec model training complete.')

    return model
