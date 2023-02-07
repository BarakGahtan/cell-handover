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
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv1d_2 = nn.Conv1d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=256,
                            num_layers=1,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 32)
        self.dense3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Raw x shape : (B, S, F) => (B, 64, 11)

        x = x.transpose(1, 2) # Shape : (B, F, S) => (B, 11, 64)

        x = self.conv1d_1(x)  # Shape : (B, F, S) == (B, C, S) // C = channel => (B, 64, 128)

        x = self.conv1d_2(x)  # Shape : (B, C, S) => (B, 128, 256)

        x = x.transpose(1, 2) # Shape : (B, S, C) == (B, S, F) => (B, 256, 128)

        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x) # Shape : (B, S, H) // H = hidden_size => (B, 64, 50)

        x = hidden[-1] # Shape : (B, H) // -1 means the last sequence => (B, 50)

        x = self.dropout(x) # Shape : (B, H) => (B, 128)

        x = self.dense1(x) # Shape : (B, 64)

        x = self.dense2(x) # Shape : (B, 32)

        x = self.dense3(x)  # Shape : (B, O) // O = output => (B, 1)
        return self.sigmoid(x)
    # def __init__(self, seq_len, number_of_features, hidden_size):
    #     super().__init__()
    #     self.layer1 = nn.Sequential(
    #         nn.Conv1d(in_channels=number_of_features, out_channels=hidden_size, kernel_size=3, stride=1),  # 11,64
    #         nn.ReLU(),
    #         # nn.MaxPool1d(4)
    #     )
    #     self.layer2 = nn.Flatten()
    #     self.layer3 = nn.Linear(in_features=64*62, out_features=512)
    #     self.layer4 = nn.Linear(in_features=512, out_features=256)
    #     self.layer5 = nn.Linear(in_features=256, out_features=1)
    #     self.sigmoid = nn.Sigmoid()
    #
    # def forward(self, x):
    #     x = x.permute(0, 2, 1) # change to (batch, seq, features)
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = self.layer5(out)
    #     # out = self.sigmoid(out)
    #     return out

#
# # class lstm_predictor(nn.Module):
# #     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, device):
# #         super(lstm_predictor, self).__init__()
# #         # Defining the number of layers and the nodes in each layer
# #         self.hidden_dim = hidden_dim
# #         self.layer_dim = layer_dim
# #         # LSTM layers
# #         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
# #         # Fully connected layer - Applies a linear transformation to the incoming data
# #         self.fc = nn.Linear(hidden_dim, output_dim)
# #         self.device = device
# #         self.activation = nn.ReLU()
# #         self.softmax = nn.Softmax()
# #         self.bn1d = nn.BatchNorm1d(num_features=int(self.hidden_dim/2))
# #
# #     def forward(self, x):
# #         # Initializing hidden state for first input with zeros
# #         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
# #         # should be 2 since D=2, N=batch size, H_out=NN size
# #         # Initializing cell state for first input with zeros
# #         c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
# #         # should be 2 since D=2, N=batch size, H_cell = hidden size
# #         # We need to detach as we are doing truncated backpropagation through time (BPTT)
# #         # If we don't, we'll backprop all the way to the start even after going through another batch
# #         # Forward propagation by passing in the input, hidden state, and cell state into the model
# #         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
# #         # should be 2 since D=2, N=batch size, H_out=NN size, h_n(D*num_layers, N,H_out) - containing the final hidden state for each element in the sequence.
# #         # c_n = (D*num_layers, N,H_cell) -  containing the final cell state for each element in the sequence.
# #         # out = self.bn1d(out)
# #         out = self.activation(out)
#
#         # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
#         # so that it can fit into the fully connected layer
#         out = out[:, -1, :]
#
#         # Convert the final state to our desired output shape (batch_size, output_dim)
#         out = self.fc(out)
#         # out = self.softmax(out) #TODO check it
#         return out
#
