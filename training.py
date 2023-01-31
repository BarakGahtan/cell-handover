# Measure our neural network by mean square error
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import optim

from load_drives import create_seq
from imblearn.over_sampling import SMOTE


def make_Tensor(array):
    # return torch.Tensor(array).float()
    return torch.from_numpy(array).float()


def balance_data_set(seq, seq_label):
    # count_label_0, count_label_1 = 0, 0
    # minority, majority = [], []
    # minority_label, majority_label = [], []
    # for i in range(len(seq)):
    #     time_series = seq[i]
    #     time_series_label = seq_label[i]
    #     if time_series_label == 0:
    #         count_label_0 = count_label_0 + 1
    #         majority.append(seq[i])
    #         majority_label.append(seq_label[i])
    #     else:
    #         count_label_1 = count_label_1 + 1
    #         minority.append(seq[i])
    #         minority_label.append(seq_label[i])
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


# def train_model(model, train_data, train_labels, lstm_flag, val_data=None, val_labels=None, num_epochs=100, verbose=10,patience=10):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('A {} device was detected.'.format(device))
#     loss_fn = torch.nn.BCEWithLogitsLoss()  #
#     optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
#     train_hist = []
#     val_hist = []
#     for t in range(num_epochs):
#         epoch_loss = 0
#         for idx, seq in enumerate(train_data):
#             if lstm_flag:
#                 model.reset_hidden_state()  # reset hidden state per seq
#             # train loss
#             seq = torch.unsqueeze(seq, 1)
#             # seq = seq.permute(1, 2, 0) # for cnn to be correctly working on all features concurrently.
#             y_pred = model(seq)
#             loss = loss_fn(y_pred[0].float(), train_labels[idx].unsqueeze(0))  # loss about 1 step
#
#             # update weights
#             optimiser.zero_grad()
#             loss.backward()
#             optimiser.step()
#
#             epoch_loss += loss.item()
#
#         train_hist.append(epoch_loss / len(train_data))
#
#         if val_data is not None:
#             with torch.no_grad():
#                 val_loss = 0
#                 for val_idx, val_seq in enumerate(val_data):
#                     model.reset_hidden_state()  # reset hidden state per seq
#                     val_seq = torch.unsqueeze(val_seq, 1)
#                     val_seq = val_seq.permute(1, 2, 0)
#                     y_val_pred = model(val_seq)
#                     val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx].unsqueeze(0))
#                     val_loss += val_step_loss
#             val_hist.append(val_loss / len(val_data))  # append in val hist
#
#             ## print loss every verbose
#             if t % verbose == 0:
#                 print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')
#
#             ## check early stopping every patience
#             if (t % patience == 0) & (t != 0):
#                 ## early stop if loss is on
#                 if val_hist[t - patience] < val_hist[t]:
#                     print('\n Early Stopping')
#                     break
#
#         elif t % verbose == 0:
#             print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')
#
#     return model, train_hist, val_hist


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        yhat = self.model(x)
        # Computes loss
        loss = self.loss_fn(y, yhat.squeeze(1)) #should check if it is good the squeeze so it will be 32 vs 32
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, device, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'./models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat.squeeze(1)).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")
        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, device,batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        # plt.close()

    def format_predictions(predictions, values, df_test, scaler):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        # df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
        return df_result
