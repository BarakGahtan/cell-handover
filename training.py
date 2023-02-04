# Measure our neural network by mean square error
import copy
import os
import pickle
from datetime import datetime
from random import choice
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook as tqdm

import architecture
from load_drives import create_seq
from imblearn.over_sampling import SMOTE


def make_Tensor(array):
    # return torch.Tensor(array).float()
    return torch.from_numpy(array).float()


def balance_data_set(seq, seq_label):
    count_label_0, count_label_1 = 0, 0
    minority, majority = [], []
    minority_label, majority_label = [], []
    for i in range(len(seq)):
        time_series = seq[i]
        time_series_label = seq_label[i]
        if time_series_label == 0:
            count_label_0 = count_label_0 + 1
            majority.append(seq[i])
            majority_label.append(seq_label[i])
        else:
            count_label_1 = count_label_1 + 1
            minority.append(seq[i])
            minority_label.append(seq_label[i])
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    seq_data_reshaped_for_balancing = seq.reshape(seq.shape[0], -1)
    oversampled_seq, oversampled_labels = sm.fit_resample(seq_data_reshaped_for_balancing, seq_label)
    oversampled_seq = oversampled_seq.reshape(oversampled_seq.shape[0], seq.shape[1], seq.shape[2])
    return oversampled_seq, oversampled_labels, count_label_0, count_label_1
    # minority = np.array(minority)
    # minority = minority.reshape(minority.shape[0],-1)
    # minority_label = np.array(minority_label)
    # minority_label = minority_label.reshape(minority_label.shape[0],-1)
    # majority = np.array(majority)
    # majority = majority.reshape(majority.shape[0],-1)
    # majority_label = np.array(majority_label)
    # majority_label = majority_label.reshape(majority_label.shape[0],-1)


def prepare_data_sets(data_frame, SEQ_LEN, balanced):
    if balanced:
        so_idxs = data_frame.index[data_frame['switchover_global'] == 1].tolist()
        no_so_idxs = [random.randint(so_idxs[0], so_idxs[-1]) for _ in range(len(so_idxs)) if random.randint(so_idxs[0], so_idxs[-1]) not in so_idxs]
        balanced_indexes = [j for i in [no_so_idxs, so_idxs] for j in i]
        random.shuffle(balanced_indexes)
        xs, ys = [], []
        for i in range(len(balanced_indexes)):
            x = data_frame.iloc[balanced_indexes[i]-SEQ_LEN:balanced_indexes[i]].drop(["switchover_global"], axis=1)
            y = data_frame["switchover_global"].iloc[balanced_indexes[i]]
            x.dropna(inplace=True)
            if len(x) == SEQ_LEN:
                xs.append(x)
                ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        data_set_size = xs.shape[0]
        train_size = int(data_set_size * 0.8)
        test_size = int(int(data_set_size - train_size) / 2)
        x_train, y_train = copy.copy(xs[:train_size]), copy.copy(ys[:train_size])
        X_val, y_val = copy.copy(xs[train_size:train_size + test_size]), copy.copy(ys[train_size:train_size + test_size])
        X_test, y_test = copy.copy(xs[train_size + test_size:]), copy.copy(ys[train_size + test_size:])
    else:
        seq, seq_label = create_seq(data_frame, SEQ_LEN)
        data_set_size = seq.shape[0]
        train_size = int(data_set_size * 0.8)
        test_size = int(int(data_set_size - train_size) / 2)
        x_train, y_train = copy.copy(seq[:train_size]), copy.copy(seq_label[:train_size])
        X_val, y_val = copy.copy(seq[train_size:train_size + test_size]), copy.copy(
            seq_label[train_size:train_size + test_size])
        X_test, y_test = copy.copy(seq[train_size + test_size:]), copy.copy(
            seq_label[train_size + test_size:])
    pickle.dump(x_train, open('x_train_balanced_64_1_imei.pkl', "wb"))
    pickle.dump(y_train, open('y_train_balanced_64_1_imei.pkl', "wb"))
    pickle.dump(X_val, open('X_val_balanced_64_1_imei.pkl', "wb"))
    pickle.dump(y_val, open('y_val_balanced_64_1_imei.pkl', "wb"))
    pickle.dump(X_test, open('X_test_balanced_64_1_imei.pkl', "wb"))
    pickle.dump(y_test, open('y_test_balanced_64_1_imei.pkl', "wb"))
    return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
        make_Tensor(y_test)
    # return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
    #     make_Tensor(y_test)
    # return x_train, y_train, X_val, y_val, X_test, y_test


def main_training_loop(epochs, training_loader, validation_loader, seq_len, number_of_features, hidden_size):
    net = architecture.cnn_lstm_hybrid(features=number_of_features)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    writer = SummaryWriter('runs/1')
    # To view, start TensorBoard on the command line with:
    #   tensorboard --logdir=runs
    # ...and open a browser tab to http://localhost:6006/
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            # basic training loop
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('Batch {}'.format(i + 1))
                # Check against the validation set
                running_vloss = 0.0

                net.train(False)  # Don't need to track gradients for validation
                for j, vdata in enumerate(validation_loader, 0):
                    vinputs, vlabels = vdata
                    voutputs = net(vinputs)
                    vloss = criterion(voutputs.squeeze(1), vlabels)
                    running_vloss += vloss.item()
                net.train(True)  # Turn gradients back on for training

                avg_loss = running_loss / 99
                avg_vloss = running_vloss / len(validation_loader)

                # Log the running loss averaged per batch
                writer.add_scalars('Training', {'Training': avg_loss}, epoch)
                writer.add_scalars('Validation Loss', {'Validation': avg_vloss}, epoch)

                running_loss = 0.0
    print('Finished Training')

    writer.flush()
