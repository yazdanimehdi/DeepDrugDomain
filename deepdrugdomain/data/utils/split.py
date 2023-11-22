"""
Utilities functions for splitting dataset taken from TDC package: https://github.com/mims-harvard/TDC/blob/main/tdc/utils/split.py and modified to fit our package structure and needs.
Thanks to the authors for sharing their code.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from typing import List, Union, Dict, Tuple
from .dataset_utils import ensure_list


def ensure_correct_frac(frac: List[float]):
    """ensure the sum of frac is 1

    Args:
        frac (list): a list of train/valid/test fractions

    Returns:
        list: a list of train/valid/test fractions
    """
    if len(frac) == 2:
        warnings.warn(
            "The length of frac is 2, we will the last one to be 1 - sum(frac)"
        )
        frac.append(1 - sum(frac))

    if len(frac) != 3:
        warnings.warn(
            "The length of frac is not 3, we will assume it is [0.8, 0.1, 0.1]")
        frac = [0.8, 0.1, 0.1]

    if sum(frac) != 1:
        warnings.warn(
            "The sum of frac is not 1, we will normalize it to 1"
        )
        frac = [f / sum(frac) for f in frac]
    return frac


def random_split(df: pd.DataFrame,
                 fold_seed: int,
                 frac: List[float]):

    train_frac, val_frac, test_frac = ensure_correct_frac(frac)
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=1
    )
    train = train_val[~train_val.index.isin(val.index)]

    return [train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)]


def cold_split(df: pd.DataFrame, fold_seed: int, frac: List[float], entities: Union[str, List[str]]):
    """create cold-split where given one or multiple columns, it first splits based on
    entities in the columns and then maps all associated data points to the partition

    Args:
            df (pd.DataFrame): dataset dataframe
            fold_seed (int): the random seed
            frac (list): a list of train/valid/test fractions
            entities (Union[str, List[str]]): either a single "cold" entity or a list of
                    "cold" entities on which the split is done

    Returns:
            dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    entities = ensure_list(entities)

    _, val_frac, test_frac = frac

    # For each entity, sample the instances belonging to the test datasets
    test_entity_instances = [
        df[e]
        .drop_duplicates()
        .sample(frac=test_frac, replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]

    # Select samples where all entities are in the test set
    test = df.copy()
    for entity, instances in zip(entities, test_entity_instances):
        test = test[test[entity].isin(instances)]

    if len(test) == 0:
        raise ValueError(
            "No test samples found. Try another seed, increasing the test frac or a "
            "less stringent splitting strategy."
        )

    # Proceed with validation data
    train_val = df.copy()
    for i, e in enumerate(entities):
        train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

    val_entity_instances = [
        train_val[e]
        .drop_duplicates()
        .sample(frac=val_frac / (1 - test_frac), replace=False, random_state=fold_seed)
        .values
        for e in entities
    ]
    val = train_val.copy()
    for entity, instances in zip(entities, val_entity_instances):
        val = val[val[entity].isin(instances)]

    if len(val) == 0:
        raise ValueError(
            "No validation samples found. Try another seed, increasing the test frac "
            "or a less stringent splitting strategy."
        )

    train = train_val.copy()
    for i, e in enumerate(entities):
        train = train[~train[e].isin(val_entity_instances[i])]

    return [
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True)]


def scaffold_split(df, seed, frac, entity):
    """create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds
    reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except:
        raise ImportError(
            "Please install rdkit by 'conda install -c conda-forge rdkit'! "
        )
    from tqdm import tqdm
    from random import Random

    from collections import defaultdict

    random = Random(seed)

    s = df[entity].values
    scaffolds = defaultdict(set)
    idx2mol = dict(zip(list(range(len(s))), s))

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except:
            warnings.warn(
                smiles + " returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    test_size = (len(df) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return [df.iloc[train].reset_index(drop=True), df.iloc[val].reset_index(drop=True), df.iloc[test].reset_index(drop=True)]
