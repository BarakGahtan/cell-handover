# Measure our neural network by mean square error
import copy

import torch
from torch import optim

from load_drives import create_seq


def make_Tensor(array):
    return torch.Tensor(array).float()
    # return torch.from_numpy(array).float()


def prepare_data_sets(data_frame, labels, SEQ_LEN):
    if labels == 0:  # data without switchover col
        seq, no_seq = create_seq(data_frame.drop(["switchover_global"], axis=1), SEQ_LEN)
    else:  # only the col we want to predict.
        seq, no_seq = create_seq(data_frame["switchover_global"], SEQ_LEN)

    data_set_size = seq.shape[0]
    train_size = int(data_set_size * 0.8)
    test_size = int(int(data_set_size - train_size) / 2)
    X_train, y_train = copy.copy(seq[:train_size]), copy.copy(no_seq[:train_size])
    X_val, y_val = copy.copy(seq[train_size:train_size + test_size]), copy.copy(
        no_seq[train_size:train_size + test_size])
    X_test, y_test = copy.copy(seq[train_size + test_size:]), copy.copy(
        no_seq[train_size + test_size:])
    # return make_Tensor(X_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
    #        make_Tensor(y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100, verbose=10,
                patience=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('A {} device was detected.'.format(device))
    loss_fn = torch.nn.L1Loss()  #
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    val_hist = []
    for t in range(num_epochs):
        epoch_loss = 0
        for idx, seq in enumerate(train_data):
            model.reset_hidden_state()  # reset hidden state per seq
            # train loss
            seq = torch.unsqueeze(seq, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx])  # loss about 1 step

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
                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])
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
