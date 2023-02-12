import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch.nn import LogSoftmax


class cnn_lstm_hybrid(nn.Module):
    def __init__(self, features):
        super(cnn_lstm_hybrid, self).__init__()
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=features,
                                  out_channels=64,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
            nn.ReLU()
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(3),
        )
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(3),
        )

        self.conv1d_4 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(3),
        )

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)

        # self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Raw x shape : (B, S, F) => (B, 64, 11)

        x = x.transpose(1, 2)  # Shape : (B, F, S) => (B, 11, 64)

        x = self.conv1d_1(x)  # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 64, 128)

        x = self.conv1d_2(x)  # Shape : (B, C, S) => (B, 128, 256)

        x = self.conv1d_3(x)  # Shape : (B, C, S) => (B, 128, 256)

        x = self.conv1d_4(x)  # Shape : (B, C, S) => (B, 128, 256)

        x = x.transpose(1, 2)  # Shape : (B, S, C) == (B, S, F) => (B, 256, 128)

        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)  # Shape : (B, S, H) // H = hidden_size => (B, 64, 50)

        x = hidden[-1]  # Shape : (B, H) // -1 means the last sequence => (B, 50)

        # x = self.dropout(x)  # Shape : (B, H) => (B, 128)

        x = self.dense1(x)  # Shape : (B, 64)

        x = self.dense2(x)  # Shape : (B, 32)

        x = self.dense3(x)  # Shape : (B, O) // O = output => (B, 1)
        return self.sigmoid(x)
