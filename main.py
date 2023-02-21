import copy
import random
import warnings
from random import choice

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import nn
import pickle

import architecture
import input_parser
import training
from load_preprocess_ds import init_drives_dataset, get_cells_per_drive_in_dataset, prepare_switchover_col, \
    preprocess_features, training_sets_init
from training import prepare_data_sets
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parsed_args = input_parser.Parser()
    opts = parsed_args.parse()
    BALANCED_FLAG = opts.bdataset
    to_balance = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opts.load_from_files == 0:
        returned_drives_by_imei_dict_train = init_drives_dataset('pickle_rick_full.pkl', opts.number_drives)
        # cells_per_drives_in_dataset_train, cells_dict_train = get_cells_per_drive_in_dataset(returned_drives_by_imei_dict_train)
        drives_by_imsi_dict = prepare_switchover_col(returned_drives_by_imei_dict_train)
        training_data = training_sets_init(drives_by_imsi_dict, opts.max_switch_over, opts.max_data_imsi)
        # correlated_data_dict_train = normalize_correlate_features(drives_by_imei_dict_train)
        correlated_data_dict_train = preprocess_features(training_data, label=opts.label)  # label = 1 = switchover, 0 = latency, 2 = loss_rate
        data_set_concat_train = pd.concat(correlated_data_dict_train, axis=0).reset_index()
        data_set_concat_train.drop(["level_0", "level_1"], axis=1, inplace=True)
        # one_hot = pd.get_dummies(data_set_concat_train['operator'], prefix='operator')  # make the operator into 1 hot encoding.
        # data_set_concat_train = pd.concat([data_set_concat_train, one_hot], axis=1)  # Concatenate the original DataFrame and the one-hot encoding
        X_train_seq, y_train_label, x_val_seq, y_val_label = \
            prepare_data_sets(data_set_concat_train, SEQ_LEN=opts.sequence_length, balanced=opts.bdataset, name=opts.model_name, label=opts.label)
        exit()
    else:  # load from saved data sets and train the model.
        if opts.label == 2:  # loss
            X_train_seq = training.make_Tensor(np.array(pickle.load(open('datasets-loss/x_train_' + opts.model_name + '.pkl', "rb"))))
            y_train_label = training.make_Tensor(np.array(pickle.load(open('datasets-loss/y_train_' + opts.model_name + '.pkl', "rb"))))
            x_val_seq = training.make_Tensor(np.array(pickle.load(open('datasets-loss/X_val_' + opts.model_name + '.pkl', "rb"))))
            y_val_label = training.make_Tensor(np.array(pickle.load(open('datasets-loss/y_val_' + opts.model_name + '.pkl', "rb"))))
        elif opts.label == 0:  # latency
            X_train_seq = training.make_Tensor(np.array(pickle.load(open('datasets-latency/x_train_' + opts.model_name + '.pkl', "rb"))))
            y_train_label = training.make_Tensor(np.array(pickle.load(open('datasets-latency/y_train_' + opts.model_name + '.pkl', "rb"))))
            x_val_seq = training.make_Tensor(np.array(pickle.load(open('datasets-latency/X_val_' + opts.model_name + '.pkl', "rb"))))
            y_val_label = training.make_Tensor(np.array(pickle.load(open('datasets-latency/y_val_' + opts.model_name + '.pkl', "rb"))))
        else:  # switch over label == 1
            X_train_seq = training.make_Tensor(np.array(pickle.load(open('datasets-so/x_train_' + opts.model_name + '.pkl', "rb"))))
            y_train_label = training.make_Tensor(np.array(pickle.load(open('datasets-so/y_train_' + opts.model_name + '.pkl', "rb"))))
            x_val_seq = training.make_Tensor(np.array(pickle.load(open('datasets-so/X_val_' + opts.model_name + '.pkl', "rb"))))
            y_val_label = training.make_Tensor(np.array(pickle.load(open('datasets-so/y_val_' + opts.model_name + '.pkl', "rb"))))
    if opts.to_train == 1:
        train_data_set = TensorDataset(X_train_seq, y_train_label)
        train_loader = DataLoader(train_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
        val_data_set = TensorDataset(x_val_seq, y_val_label)
        val_loader = DataLoader(val_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
        features_count = X_train_seq.shape[2]
        training_class = training.optimizer(opts.model_name, opts.epoch_number, train_loader, val_loader, val_loader, opts.sequence_length,
                                            features_count, opts.neuralnetwork_size, opts.learn_rate, opts.batch_size, opts.label)
        if opts.label == 0 or opts.label == 2:
            training_class.main_training_loop_latency()
        else:
            training_class.main_training_loop_switchover()
        print("Finished training model " + opts.model_name + "_" + str(opts.batch_size))
    else:  # testing the model.
        # x_test_seq = training.make_Tensor(np.array(pickle.load(open('datasets-latency/x_train_seq_128_80_label_0' + '.pkl', "rb"))))
        # y_test_label = training.make_Tensor(np.array(pickle.load(open('datasets-latency/y_train_seq_128_80_label_0' + '.pkl', "rb"))))
        x_test_seq = training.make_Tensor(np.array(pickle.load(open('x_train_' + opts.model_name + '.pkl', "rb"))))
        y_test_label = training.make_Tensor(np.array(pickle.load(open('y_train_' + opts.model_name + '.pkl', "rb"))))
        test_loader = DataLoader(TensorDataset(x_test_seq, y_test_label), batch_size=512, shuffle=False, drop_last=True)
        model = architecture.cnn_lstm_hybrid(features=x_test_seq.shape[2], label=0)
        model.load_state_dict(torch.load('best_model_seq_64_80_label_2_batch_size_512.pt'))
        training.test_model_mse(test_loader=test_loader, given_model=model, opts=opts)
        print("Finished testing model " + opts.model_name + "_" + str(opts.batch_size))
        print("finished making a data set.")