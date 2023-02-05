import copy
import random
import warnings
from random import choice

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import nn
import pickle
import input_parser
import training
from load_drives import init_drives_dataset, get_cells_per_drive_in_dataset, prepare_switchover_col, \
    normalize_correlate_features, training_sets_init
from training import prepare_data_sets
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parsed_args = input_parser.Parser()
    opts = parsed_args.parse()
    NUM_DRIVES = opts.number_drives
    DRIVE_NUM_TRAIN = opts.starting_drive_train
    SEQ_LEN = opts.sequence_length
    NN_SIZE = opts.neuralnetwork_size
    NN_LAYERS = opts.neuralnetwork_layers
    BALANCED_FLAG = opts.bdataset
    batch_size = opts.batch_size
    n_epochs = opts.epoch_number
    to_balance = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opts.load_from_files == 0:
        returned_drives_by_imei_dict_train = init_drives_dataset('pickle_rick.pkl', DRIVE_NUM_TRAIN, NUM_DRIVES)
        # cells_per_drives_in_dataset_train, cells_dict_train = get_cells_per_drive_in_dataset(returned_drives_by_imei_dict_train)
        drives_by_imei_dict_train = prepare_switchover_col(returned_drives_by_imei_dict_train)
        training_data = training_sets_init(drives_by_imei_dict_train, opts.max_switch_over)
        # correlated_data_dict_train = normalize_correlate_features(drives_by_imei_dict_train)
        correlated_data_dict_train = normalize_correlate_features(training_data)
        data_set_concat_train = pd.concat(correlated_data_dict_train, axis=0).reset_index()
        data_set_concat_train.drop(["level_0", "level_1"], axis=1, inplace=True)
        X_train_seq, y_train_label, x_val_seq, y_val_label, x_test_seq, y_test_label = \
            prepare_data_sets(data_set_concat_train, SEQ_LEN=SEQ_LEN, balanced=to_balance, name=opts.model_name)
    else:
        X_train_seq = training.make_Tensor(np.array(pickle.load(open('x_train_balanced_64_all_imei.pkl', "rb"))))
        y_train_label = training.make_Tensor(np.array(pickle.load(open('y_train_balanced_64_all_imei.pkl', "rb"))))
        x_val_seq = training.make_Tensor(np.array(pickle.load(open('X_val_balanced_64_all_imei.pkl', "rb"))))
        y_val_label = training.make_Tensor(np.array(pickle.load(open('y_val_balanced_64_all_imei.pkl', "rb"))))
        x_test_seq = training.make_Tensor(np.array(pickle.load(open('X_test_balanced_64_all_imei.pkl', "rb"))))
        y_test_label = training.make_Tensor(np.array(pickle.load(open('y_test_balanced_64_all_imei.pkl', "rb"))))

    # seq = pickle.load(open('x_data_eli1.pkl', "rb"))
    # y_data = pickle.load(open('y_data_eli1.pkl', "rb"))
    # seq_label = np.array(LabelEncoder().fit_transform(y_data)).astype("float32")
    # seq = np.array(seq)
    # data_set_size = seq.shape[0]
    # train_size = int(data_set_size * 0.8)
    # test_size = int(int(data_set_size - train_size) / 2)
    # x_train, y_train = copy.copy(seq[:train_size]), copy.copy(seq_label[:train_size])
    # X_val, y_val = copy.copy(seq[train_size:train_size + test_size]), copy.copy(
    #     seq_label[train_size:train_size + test_size])
    # X_test, y_test = copy.copy(seq[train_size + test_size:]), copy.copy(
    #     seq_label[train_size + test_size:])
    # X_train_seq = training.make_Tensor(x_train)
    # y_train_label = training.make_Tensor(y_train)
    # x_val_seq = training.make_Tensor(X_val)
    # y_val_label = training.make_Tensor(y_val)
    # x_test_seq = training.make_Tensor(X_test)
    # y_test_label = training.make_Tensor(y_test)

    if opts.to_train == 1:
        train_data_set = TensorDataset(X_train_seq, y_train_label)
        train_loader = DataLoader(train_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
        val_data_set = TensorDataset(x_val_seq, y_val_label)
        val_loader = DataLoader(val_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
        test_data_set = TensorDataset(x_test_seq, y_test_label)
        test_loader = DataLoader(test_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
        features_count = X_train_seq.shape[2]
        training_class = training.optimizer(opts.name, n_epochs, train_loader, val_loader, test_loader, SEQ_LEN, features_count, NN_SIZE,
                                            opts.learn_rate)
        training_class.main_training_loop()
    else:
        print("finished making a data set.")

    # cnn_model = cnn1d_model(seq_len=SEQ_LEN, number_of_features=features_count)  # number features is the seqeunce len * max pooling of Conv1D
    # combined_model = architecture.cnn_lstm_combined(cnn_model, number_features=features_count,
    #                                                 n_hidden=NN_LAYERS, seq_len=SEQ_LEN,
    #                                                 n_layers=NN_LAYERS, cnn_enable=0)  # seq_len - delta t window to look back.
    # # model, train_hist, val_hist = train_model(combined_model, X_train_seq, y_train_label, val_data=x_val_seq,
    #                                           val_labels=y_val_label, lstm_flag=LSTM_FLAG)
    # architecture.plot_train(train_hist, val_hist)

    # learning_model.test_model(x_test_seq, y_test_label, model)

    # SET TEST DATA
    # drives_by_modem_test, returned_drives_by_imei_dict_test = init_drives_dataset('pickle_rick.pkl', DRIVE_NUM_TEST,
    #                                                                               NUM_DRIVES)
    # cells_per_drives_in_dataset_test, cells_dict_test = get_cells_per_drive_in_dataset(
    #     returned_drives_by_imei_dict_test)
    # drives_by_imei_dict_test = prepare_switchover_col(returned_drives_by_imei_dict_test)
    # correlated_data_dict_test = normalize_correlate_features(drives_by_imei_dict_test)
    # data_set_concat_test = pd.concat(correlated_data_dict_test, axis=0).reset_index()
    # data_set_concat_test = data_set_concat_test.drop(["level_0", "level_1"], axis=1,
    #                                                  inplace=True)  # should go into 1D-CNN MODEL
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # drives_by_imei_dict, rsrp_dictionary = prepare_distance_to_cells(drives_by_imei_dict, cells_dict)
    # visualize_drives(returned_drives_by_imei_dict_train, cells_per_drives_in_dataset_train)
    # build_regression_model(drives_by_imei_dict_train)
