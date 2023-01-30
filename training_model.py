# Measure our neural network by mean square error
import copy

import numpy as np
import torch
from torch import optim

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
    return oversampled_seq, oversampled_labels
    # minority = np.array(minority)
    # minority = minority.reshape(minority.shape[0],-1)
    # minority_label = np.array(minority_label)
    # minority_label = minority_label.reshape(minority_label.shape[0],-1)
    # majority = np.array(majority)
    # majority = majority.reshape(majority.shape[0],-1)
    # majority_label = np.array(majority_label)
    # majority_label = majority_label.reshape(majority_label.shape[0],-1)


def prepare_data_sets(data_frame, SEQ_LEN, balanced):
    seq, seq_label = create_seq(data_frame, SEQ_LEN)
    oversampled_seq, oversampled_labels = balance_data_set(seq, seq_label)
    if balanced:
        data_set_size = oversampled_seq.shape[0]
        train_size = int(data_set_size * 0.8)
        test_size = int(int(data_set_size - train_size) / 2)
        x_train, y_train = copy.copy(oversampled_seq[:train_size]), copy.copy(oversampled_labels[:train_size])
        X_val, y_val = copy.copy(oversampled_seq[train_size:train_size + test_size]), copy.copy(
            oversampled_labels[train_size:train_size + test_size])
        X_test, y_test = copy.copy(oversampled_seq[train_size + test_size:]), copy.copy(
            oversampled_labels[train_size + test_size:])
    else:
        data_set_size = seq.shape[0]
        train_size = int(data_set_size * 0.8)
        test_size = int(int(data_set_size - train_size) / 2)
        x_train, y_train = copy.copy(seq[:train_size]), copy.copy(seq_label[:train_size])
        X_val, y_val = copy.copy(seq[train_size:train_size + test_size]), copy.copy(
            seq_label[train_size:train_size + test_size])
        X_test, y_test = copy.copy(seq[train_size + test_size:]), copy.copy(
            seq_label[train_size + test_size:])
    return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
        make_Tensor(y_test)
    # return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
    #     make_Tensor(y_test)
    # return x_train, y_train, X_val, y_val, X_test, y_test


def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100, verbose=10,
                patience=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('A {} device was detected.'.format(device))
    loss_fn = torch.nn.BCEWithLogitsLoss()  #
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    val_hist = []
    for t in range(num_epochs):
        epoch_loss = 0
        for idx, seq in enumerate(train_data):
            model.reset_hidden_state()  # reset hidden state per seq
            # train loss
            seq = torch.unsqueeze(seq, 1)
            seq = seq.permute(1, 2, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx].unsqueeze(0))  # loss about 1 step

            # update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        train_hist.append(epoch_loss / len(train_data))

        if val_data is not None:
            with torch.no_grad():
                val_loss = 0
                for val_idx, val_seq in enumerate(val_data):
                    model.reset_hidden_state()  # reset hidden state per seq
                    val_seq = torch.unsqueeze(val_seq, 1)
                    val_seq = val_seq.permute(1, 2, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx].unsqueeze(0))
                    val_loss += val_step_loss
            val_hist.append(val_loss / len(val_data))  # append in val hist

            ## print loss every verbose
            if t % verbose == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

            ## check early stopping every patience
            if (t % patience == 0) & (t != 0):
                ## early stop if loss is on
                if val_hist[t - patience] < val_hist[t]:
                    print('\n Early Stopping')
                    break

        elif t % verbose == 0:
            print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')

    return model, train_hist, val_hist
