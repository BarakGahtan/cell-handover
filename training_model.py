# Measure our neural network by mean square error
import copy

import torch
from torch import optim

from load_drives import create_seq


def make_Tensor(array):
    return torch.Tensor(array).float()


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


def train(model, training_count, data_set_train, output):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('A {} device was detected.'.format(device))

    x = torch.tensor(data_set_train, dtype=torch.float, device=device)
    y = torch.tensor(output, dtype=torch.float, device=device)
    criterion = torch.nn.MSELoss()
    # Train our network with a simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Train our network a using the entire dataset 5 times
    for epoch in range(training_count):
        totalLoss = 0
        for i in range(len(x)):
            # Single Forward Pass
            ypred = model(x[i])

            # Measure how well the model predicted vs the actual value
            loss = criterion(ypred, y[i])

            # Track how well the model predicted (called loss)
            totalLoss += loss.item()

            # Update the neural network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print out our loss after each training iteration
        print("Total Loss: ", totalLoss)
