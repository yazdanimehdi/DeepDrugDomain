import pickle

from dataset import DTIData
from data_preprocess import human_process, celegans_process
import os


def build_dataset(config):
    if config.dataset == 'kiba':
        pass
    elif config.dataset == 'davis':
        pass
    elif config.dataset == 'bindingdb':
        pass
    elif config.dataset == 'ibmbindingdb':
        pass
    elif config.dataset == 'dude':
        pass
    elif config.dataset == 'human':
        df_dir = human_process(config)
        if os.path.exists(config.processed_file_dir + 'human.pkl'):
            with open(config.processed_file_dir + 'human.pkl', 'rb') as fp:
                p_graph, s_graph = pickle.load(fp)
        else:
            p_graph = None
            s_graph = None
        data_train = DTIData('human', df_dir['train'], config.processed_file_dir, config.pdb_dir, p_graph, s_graph)
        data_val = DTIData('human', df_dir['val'], config.processed_file_dir, config.pdb_dir, p_graph, s_graph)
        data_test = DTIData('human', df_dir['test'], config.processed_file_dir, config.pdb_dir, p_graph, s_graph)
        return data_train, data_val, data_test

    elif config.dataset == 'celegans':
        df_dir = celegans_process(config)
        if os.path.exists(config.processed_file_dir + 'celegans.pkl'):
            with open(config.processed_file_dir + 'celegans.pkl', 'rb') as fp:
                p_graph, s_graph = pickle.load(fp)
        else:
            p_graph = None
            s_graph = None
        data_train = DTIData('celegans', df_dir['train'], config.processed_file_dir, config.pdb_dir, p_graph, s_graph)
        data_val = DTIData('celegans', df_dir['val'], config.processed_file_dir, config.pdb_dir, p_graph, s_graph)
        data_test = DTIData('celegans', df_dir['test'], config.processed_file_dir, config.pdb_dir, p_graph, s_graph)
        return data_train, data_val, data_test