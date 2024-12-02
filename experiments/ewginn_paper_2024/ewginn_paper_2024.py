import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params
import argparse
from graphinstructed.layers import EdgeWiseGraphInstructed, GraphInstructed
from graphinstructed.utils import dict2sparse
from sklearn.preprocessing import StandardScaler
import yaml
from yaml import Loader
import pickle as pkl
import os


def mre_av(y_true, y_pred):
    phi_true = tf.reduce_sum(y_true, axis=-1)
    mean_relerr = tf.reduce_mean(
        tf.math.abs(y_true - y_pred) / tf.expand_dims(phi_true, axis=-1),
        axis=-1
    )
    return tf.reduce_mean(mean_relerr)


def mre_phi(y_true, y_pred):
    phi_pred = tf.reduce_sum(y_pred, axis=-1)
    phi_true = tf.reduce_sum(y_true, axis=-1)

    phi_relerr = tf.math.abs(phi_pred - phi_true) / phi_true

    return tf.reduce_mean(phi_relerr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=str, help='Options: BA or ER')
    parser.add_argument('gi_type', type=str, help='Options: GI or EWGI')
    parser.add_argument('-ci', '--config_index', type=int, default=0, help='minimum 0, maximum 59')
    parser.add_argument('-rs', '--random_seed', type=int, default=None)
    parser.add_argument('-s', '--save', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    graph_folder = f'experiments/ewginn_paper_2024/data/graph{args.graph}'
    data_folder = f'{graph_folder}/experiment_split'
    results_folder = f'experiments/ewginn_paper_2024/results/graph{args.graph}'
    configs_filepath = f'experiments/ewginn_paper_2024/configs/configs_{args.graph}.yml'
    config_index = args.config_index

    Xtrain = pd.read_pickle(f'{data_folder}/Xtrain.pkl')
    Xval = pd.read_pickle(f'{data_folder}/Xval.pkl')
    Xtest = pd.read_pickle(f'{data_folder}/Xtest.pkl')

    Ytrain = pd.read_pickle(f'{data_folder}/Ytrain.pkl')
    Yval = pd.read_pickle(f'{data_folder}/Yval.pkl')
    Ytest = pd.read_pickle(f'{data_folder}/Ytest.pkl')

    stdscal = StandardScaler()
    stdscal.fit(Xtrain.values)

    Xtrain_preproc = stdscal.transform(Xtrain.values)
    Xval_preproc = stdscal.transform(Xval.values)
    Xtest_preproc = stdscal.transform(Xtest.values)

    Ytrain_preproc = np.abs(Ytrain.values)
    Yval_preproc = np.abs(Yval.values)
    Ytest_preproc = np.abs(Ytest.values)

    with open(configs_filepath, 'r') as file:
        cfgs_list = yaml.load(file, Loader=Loader)
        config = cfgs_list[config_index]

    config['graph'] = args.graph
    config['model'] = f'{args.gi_type}NN'
    config['config_ID'] = config_index
    if args.random_seed is not None:
        config['random_state'] = args.random_seed
    config['batch_size'] = 32
    config['max_epochs'] = 5000
    config['learning_rate'] = 0.002
    config['optimizer_class'] = 'Adam'
    config['loss_function'] = 'mse'
    config['early_stopping_startingepoch'] = 200
    config['early_stopping_patience'] = 550
    config['reduce_lr_on_plateau_patience'] = 50
    config['reduce_lr_on_plateau_factor'] = 0.5

    np.random.seed(config['random_state'])
    tf.random.set_seed(config['random_state'])
    tf.keras.utils.set_random_seed(
        config['random_state']
    )

    # COMPLETE REPRODUCIBILITY IS NOT POSSIBLE BECAUSE:
    # A deterministic GPU implementation of SparseTensorDenseMatmulOp is not currently available in TensorFlow.
    #
    # THEN, WE CANNOT ACTIVATE DETERMINISTIC OPERATIONS WHILE USING GPUs FOR TRAINING (SEE BELOW).
    # # If using TensorFlow, this will make GPU ops as deterministic as possible, but it will affect the overall
    # # performance, so be mindful of that.
    # tf.config.experimental.enable_op_determinism()

    # GET GRAPH
    with open(f'{graph_folder}/linegraph_mysparse_dict.yml', 'r') as file:
        Asparse = dict2sparse(yaml.load(file, Loader=Loader))
    with open(f'{graph_folder}/info_dict.yml', 'r') as file:
        info_dict = yaml.load(file, Loader=Loader)

    out_nodes = [int(c.replace('F', '')) for c in Ytrain.columns.tolist()]

    hidden_layer_config = {
        'adj_mat': Asparse,
        'num_filters': config['filters'],
        'activation': 'linear',  # THE CHOSEN ACTIVATION IS APPLIED AFTER BATCH-NORMALIZATION; SEE BELOW.
        'kernel_initializer': config['kernel_initializer'],
    }

    out_layer_config = {
        'adj_mat': Asparse[:, out_nodes],
        'colkeys': out_nodes,
        'num_filters': config['filters'],
        'activation': 'linear',
        'kernel_initializer': config['kernel_initializer'],
        'pool': config['pooling'],
    }

    if args.gi_type == 'EWGI':
        GI = EdgeWiseGraphInstructed
    else:
        GI = GraphInstructed

    # build a model
    I_layer = tf.keras.layers.Input(Xtrain_preproc.shape[1])
    layers_list = [I_layer]
    for h in range(config['depth']):
        layers_list.append(
            GI(**hidden_layer_config)(layers_list[-1])
        )
        layers_list.append(
            tf.keras.layers.BatchNormalization()(layers_list[-1])
        )
        layers_list.append(
            tf.keras.layers.Activation(config['activation'])(layers_list[-1])
        )

    layers_list.append(
        GI(**out_layer_config)(layers_list[-1])
    )

    model = tf.keras.models.Model(inputs=layers_list[0], outputs=layers_list[-1])

    # COMPILE THE MODEL
    optimizer = getattr(tf.keras.optimizers, config['optimizer_class'])(learning_rate=config['learning_rate'])

    model.compile(optimizer=optimizer,
                  loss=config['loss_function'],
                  metrics=[mre_av, mre_phi, 'mae']
                  )

    history = model.fit(x=Xtrain_preproc,
                        y=Ytrain_preproc,
                        epochs=config['max_epochs'],
                        batch_size=config['batch_size'],
                        validation_data=(Xval_preproc, Yval_preproc),
                        callbacks=[
                            tf.keras.callbacks.TerminateOnNaN(),
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             patience=config['early_stopping_patience'],
                                                             restore_best_weights=True,
                                                             verbose=True,
                                                             start_from_epoch=config['early_stopping_startingepoch']
                                                             ),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                 patience=config['reduce_lr_on_plateau_patience'],
                                                                 factor=config['reduce_lr_on_plateau_factor'],
                                                                 min_lr=1e-6,
                                                                 verbose=True
                                                                 )
                        ],
                        verbose=True
                        )

    test_eval = model.evaluate(Xtest_preproc, Ytest_preproc,
                               verbose=False,
                               batch_size=Xtest_preproc.shape[0],
                               return_dict=True
                               )

    config['trainable_params'] = count_params(model.trainable_weights)
    config['non_trainable_params'] = count_params(model.non_trainable_weights)
    config['total_params'] = count_params(model.weights)

    if args.save:
        save_folder = f'{results_folder}/cfg{str(args.config_index).zfill(3)}_{args.gi_type}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(f'{save_folder}/history_rs{config["random_state"]}.pkl', 'wb') as file:
            pkl.dump(history.history, file)
        with open(f'{save_folder}/testperf_rs{config["random_state"]}.yml', 'w') as file:
            yaml.dump(test_eval, file)
        with open(f'{save_folder}/config_rs{config["random_state"]}.yml', 'w') as file:
            yaml.dump(config, file)
        model.export(f'{save_folder}/{args.gi_type}NNartifact_cfg{str(args.config_index).zfill(3)}_rs{config["random_state"]}')

        summary_df = f'{results_folder}/{args.graph}_summary_results.csv'
        config.update(test_eval)
        if not os.path.exists(summary_df):
            sumdf = pd.DataFrame(config, index=[0])
        else:
            sumdf_old = pd.read_csv(summary_df, index_col=0)
            sumdf = pd.DataFrame(config, index=[0])
            sumdf = pd.concat([sumdf_old, sumdf], axis=0, ignore_index=True)

        sumdf.to_csv(summary_df)




