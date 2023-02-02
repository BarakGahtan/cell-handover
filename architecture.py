import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch.nn import LogSoftmax


class cnn_predictor(nn.Module):
    def __init__(self, seq_len, number_of_features, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(number_of_features, hidden_size, kernel_size=seq_len, stride=1),  # 14,5
            nn.ReLU(),
        )
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1) # change to (batch, seq, features)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.sigmoid(out)
        return out

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
