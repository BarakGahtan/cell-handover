import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch.nn import LogSoftmax


class cnn1d_model(nn.Module):
    def __init__(self, seq_len, number_of_features):
        super().__init__()
        # self.layer1 = nn.Conv1d(1, 64, kernel_size=3, stride=1)
        self.logSoftmax = LogSoftmax(dim=1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(number_of_features, 64, kernel_size=seq_len, stride=1),  # 14,5
            nn.ReLU(),
            # nn.MaxPool1d(5)
        )
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.MaxPool1d(5)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out.unsqueeze(2))
        # out = self.logSoftmax(out)
        return out


class cnn_lstm_combined(nn.Module):
    def __init__(self, number_features, n_hidden, seq_len, n_layers, cnn_enable):
        super(cnn_lstm_combined, self).__init__()
        self.cnn_enable = cnn_enable
        self.hidden = None
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        # self.c1 = model
        self.lstm = nn.LSTM(
            input_size=number_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, 1, self.n_hidden),
            torch.zeros(self.n_layers, 1, self.n_hidden)
        )
        # moved the middle index to be 1 instead of seq length.

    def forward(self, sequences):
        # if self.cnn_enable:
        #     sequences = self.c1(sequences)
        lstm_out, self.hidden = self.lstm(sequences.unsqueeze(0).flatten(-2), self.hidden)  # making it into (1-batch, seq-time, features)
        last_time_step = lstm_out.flatten(-2)  # take all of the output cells
        y_pred = self.linear(last_time_step)  # there should be no activation in that layer because we use bncross entropy
        return y_pred


class TimeSeriesLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device):
        super(TimeSeriesLSTMModel, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = device
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

def plot_train(train_hist, val_hist):
    plt.plot(train_hist, label="Training loss")
    plt.plot(val_hist, label="Val loss")
    plt.legend()
    plt.show()


def test_model(x_test_seq, y_test_label, model):
    pred_dataset = x_test_seq
    with torch.no_grad():
        preds = []
        for _ in range(len(pred_dataset)):
            model.reset_hidden_state()
            y_test_pred = model(torch.unsqueeze(pred_dataset[_], 0))
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
    plt.plot(np.array(y_test_label), label='True')
    plt.plot(np.array(preds), label='Pred')
    plt.legend()
