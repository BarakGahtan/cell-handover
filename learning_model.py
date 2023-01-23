import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class cnn_extractor(nn.Module):
    def __init__(self, n_features=None):
        super().__init__()
        # self.layer1 = nn.Conv1d(1, 64, kernel_size=3, stride=1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(3))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(3))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out.unsqueeze(1))
        return out


class cnn_lstm_combined(nn.Module):
    def __init__(self, model, number_features, n_hidden, seq_len, n_layers):
        super(cnn_lstm_combined, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.c1 = model
        self.lstm = nn.LSTM(
            input_size=64*63,
            hidden_size=n_hidden,
            num_layers=n_layers
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len - 1, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len - 1, self.n_hidden)
        )

    def forward(self, sequences):
        # sequences = self.c1(sequences.view(len(sequences), 1, -1))
        sequences = self.c1(sequences)
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len - 1, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len - 1, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
