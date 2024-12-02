import numpy as np
import pandas as pd
import yaml
from yaml import Loader
import pickle as pkl
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=str, help='Options: BA or ER')
    parser.add_argument('-rs', '--random_seed', type=int, default=1400)
    parser.add_argument('-tr', '--training_cardinality', type=int, default=400)
    parser.add_argument('-val', '--validation_cardinality', type=int, default=100)
    parser.add_argument('-tst', '--test_cardinality', type=int, default=3000)
    parser.add_argument('-s', '--save', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    graph_data_folder = f'experiments/ewginn_paper_2024/data/graph{args.graph}'

    with open(f'{graph_data_folder}/info_dict.yml', 'r') as file:
        info_dict = yaml.load(file, Loader=Loader)

    XY_df = pd.read_pickle(f'{graph_data_folder}/capflux_df.pkl')

    np.random.seed(args.random_seed)

    T = args.training_cardinality
    V = args.validation_cardinality
    P = args.test_cardinality

    target_nodes = info_dict['neig_SINK_edge_inds']
    target_nodes.sort()

    feature_cols = info_dict['cap_desc'].columns.tolist()
    target_cols = [f'F{ii}' for ii in target_nodes]

    ii_tot = np.random.permutation(XY_df.index.tolist())
    ii_T = ii_tot[:T]
    ii_V = ii_tot[T:T + V]
    if P is None:
        ii_P = ii_tot[T + V:]
        P = len(ii_P)
    else:
        P = P
        ii_P = ii_tot[T + V:]
        ii_P = ii_P[-P:]

    Xtrain = XY_df.loc[ii_T, feature_cols]
    Xval = XY_df.loc[ii_V, feature_cols]
    Xtest = XY_df.loc[ii_P, feature_cols]

    Ytrain = XY_df.loc[ii_T, target_cols]
    Yval = XY_df.loc[ii_V, target_cols]
    Ytest = XY_df.loc[ii_P, target_cols]

    if args.save:
        generation_options = {'random_seed': args.random_seed,
                              'training_cardinality': T,
                              'validation_cardinality': V,
                              'test_cardinality': P
                              }
        with open(f'{graph_data_folder}/experiment_split/generation_options.yml', 'w') as file:
            yaml.dump(generation_options, file)

        indices = {'ii_T': ii_T, 'ii_V': ii_V, 'ii_P': ii_P}
        with open(f'{graph_data_folder}/experiment_split/split_indices.pkl', 'wb') as file:
            pkl.dump(indices, file)

        Xtrain.to_pickle(f'{graph_data_folder}/experiment_split/Xtrain.pkl')
        Xval.to_pickle(f'{graph_data_folder}/experiment_split/Xval.pkl')
        Xtest.to_pickle(f'{graph_data_folder}/experiment_split/Xtest.pkl')

        Ytrain.to_pickle(f'{graph_data_folder}/experiment_split/Ytrain.pkl')
        Yval.to_pickle(f'{graph_data_folder}/experiment_split/Yval.pkl')
        Ytest.to_pickle(f'{graph_data_folder}/experiment_split/Ytest.pkl')



