from random import shuffle

import pandas as pd

from config import Config


def celegans_process(config: Config):
    df = pd.read_csv(config.raw_data_dir + '/celegans/' + "celegansSeqPdb.txt")
    with open(config.raw_data_dir + '/celegans/' + "true.txt", 'r') as fp:
        train_raw = fp.read()
    raw_data = train_raw.split("\n")
    shuffle(raw_data)
    raw_data_train = raw_data[0: int(len(raw_data) * config.train_split)]
    raw_data_valid = raw_data[int(len(raw_data) * config.train_split): int(
        len(raw_data) * (config.train_split + config.val_split))]
    raw_data_test = raw_data[int(len(raw_data) * (config.train_split + config.val_split)): int(len(raw_data))]
    train_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for item in raw_data_train:
        try:
            a = item.split()
            smile = a[0]
            sequence = a[1]
            pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
            if pdb_code is not None:
                label = 1 if a[2] == '1' else 0
                train_df = train_df.append({'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label},
                                ignore_index=True)
        except:
            pass

    val_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for item in raw_data_valid:
        try:
            a = item.split()
            smile = a[0]
            sequence = a[1]
            pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
            if pdb_code is not None:
                label = 1 if a[2] == '1' else 0
                val_df = val_df.append(
                    {'SMILE': smile, 'PDB': pdb_code, 'TargetSequence': sequence, 'Label': label},
                    ignore_index=True)
        except:
            pass

    test_df = pd.DataFrame(columns=['SMILE', 'PDB', 'TargetSequence', 'Label'])
    for item in raw_data_test:
        try:
            a = item.split()
            smile = a[0]
            sequence = a[1]
            pdb_code = df.loc[df["sequence"] == sequence]["pdb_id"].item()[0:4]
            if pdb_code is not None:
                label = 1 if a[2] == '1' else 0
                test_df = test_df.append(
                    {'SMILE': smile, 'PDB': pdb_code[0:4], 'TargetSequence': sequence, 'Label': label},
                    ignore_index=True)
        except:
            pass
    train_df_dir = config.df_dir + 'celegans_train' + '.csv'
    test_df_dir = config.df_dir + 'celegans_val' + '.csv'
    val_df_dir = config.df_dir + 'celegans_test' + '.csv'
    train_df.to_csv(train_df_dir)
    val_df.to_csv(val_df_dir)
    test_df.to_csv(test_df_dir)
    return {'train': train_df_dir, 'val': val_df_dir, 'test': test_df_dir}