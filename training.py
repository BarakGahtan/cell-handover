# Measure our neural network by mean square error
import copy
import os
import pickle
import time
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


def prepare_data_sets(data_frame, SEQ_LEN, balanced, name):
    if balanced:
        so_idxs = data_frame.index[data_frame['switchover_global'] == 1].tolist()
        no_so_idxs = [random.randint(so_idxs[0], so_idxs[-1]) for _ in range(len(so_idxs)) if random.randint(so_idxs[0], so_idxs[-1]) not in so_idxs]
        balanced_indexes = [j for i in [no_so_idxs, so_idxs] for j in i]
        random.shuffle(balanced_indexes)
        xs, ys = [], []
        for i in range(len(balanced_indexes)):
            x = data_frame.iloc[balanced_indexes[i] - SEQ_LEN:balanced_indexes[i]].drop(["switchover_global"], axis=1)
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
    pickle.dump(x_train, open('x_train_' + name + '.pkl', "wb"))
    pickle.dump(y_train, open('y_train_' + name + '.pkl', "wb"))
    pickle.dump(X_val, open('X_val_' + name + '.pkl', "wb"))
    pickle.dump(y_val, open('y_val_' + name + '.pkl', "wb"))
    pickle.dump(X_test, open('X_test_' + name + '.pkl', "wb"))
    pickle.dump(y_test, open('y_test_' + name + '.pkl', "wb"))
    return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
        make_Tensor(y_test)
    # return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
    #     make_Tensor(y_test)
    # return x_train, y_train, X_val, y_val, X_test, y_test


class optimizer:
    def __init__(self, name, epochs, training_loader, validation_loader, test_loader, seq_len, number_of_features, hidden_size, learning_rate, batch_size):
        self.name = name
        self.train_loader = training_loader
        self.epochs = epochs
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.sequence_len = seq_len
        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.net = architecture.cnn_lstm_hybrid(features=number_of_features)
        self.learn_rate = learning_rate
        self.average_loss_validation, self.average_loss_training, self.average_loss_test = [], [], []
        self.avg_accuracy_prediction_1 = []
        self.avg_accuracy_prediction_2 = []
        self.avg_accuracy_prediction_3 = []
        self.avg_accuracy_prediction_4 = []
        self.avg_accuracy_prediction_5 = []
        self.avg_accuracy_prediction_1_test = []
        self.avg_accuracy_prediction_2_test = []
        self.avg_accuracy_prediction_3_test = []
        self.avg_accuracy_prediction_4_test = []
        self.avg_accuracy_prediction_5_test = []
        self.epoch_number = []
        self.time_diff = 0
        self.batch_size = batch_size
        self.writer = SummaryWriter('models/' + self.name + '_batch_size_' + str(self.batch_size))

    def write_to_file(self):
        df = pd.DataFrame({'avg_validation_loss': self.average_loss_validation,
                           'avg_training_loss': self.average_loss_training,
                           'accuracy_05_val': self.avg_accuracy_prediction_1,
                           'accuracy_055_val': self.avg_accuracy_prediction_2,
                           'accuracy_06_val': self.avg_accuracy_prediction_3,
                           'accuracy_065_val': self.avg_accuracy_prediction_4,
                           'accuracy_07_val': self.avg_accuracy_prediction_5,
                           'epochs': self.epoch_number,
                           'training time': self.time_diff,
                           'tl_samples_batches_count': len(self.train_loader),
                           'tl_sample_count' : len(self.train_loader.dataset),
                           'vl_samples_batches_count': len(self.validation_loader),
                           'vl_sample_count': len(self.validation_loader.dataset),
                           'testl_samples_batches_count': len(self.test_loader),
                           'testl_sample_count': len(self.test_loader.dataset),
                           'avg_test_loss': self.average_loss_test,
                           'accuracy_05_test': self.avg_accuracy_prediction_1_test,
                           'accuracy_055_test': self.avg_accuracy_prediction_2_test,
                           'accuracy_06_test': self.avg_accuracy_prediction_3_test,
                           'accuracy_065_test': self.avg_accuracy_prediction_4_test,
                           'accuracy_07_test': self.avg_accuracy_prediction_5_test},
                          )
        df.to_csv(self.name +'_batch_size_' + str(self.batch_size) + '.csv', index=False)

    def main_training_loop(self):
        avg_vloss = float('inf')
        criterion = torch.nn.BCELoss()
        optim_to_learn = optim.Adam(self.net.parameters(), lr=self.learn_rate)
        best_val_loss = float('inf')
        counter = 0
        patience = 20
        # To view, start TensorBoard on the command line with:
        #   tensorboard --logdir=model/sseq_32_20
        # ...and open a browser tab to http://localhost:6006/
        start_time = time.time()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                optim_to_learn.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                loss.backward()
                optim_to_learn.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print('Batch {}'.format(i + 1))
                    running_vloss, running_true_accuracy = 0.0, 0.0
                    with torch.no_grad():
                        self.net.train(False)  # Don't need to track gradients for validation
                        for j, vdata in enumerate(self.test_loader, 0):
                            vinputs, vlabels = vdata
                            voutputs = self.net(vinputs)
                            vloss = criterion(voutputs.squeeze(1), vlabels)
                            running_vloss += vloss.item()
                            vector_1 = np.where(voutputs.numpy().squeeze(1) > 0.5, True, False)
                            vector_2 = np.where(voutputs.numpy().squeeze(1) > 0.55, True, False)
                            vector_3 = np.where(voutputs.numpy().squeeze(1) > 0.6, True, False)
                            vector_4 = np.where(voutputs.numpy().squeeze(1) > 0.65, True, False)
                            vector_5 = np.where(voutputs.numpy().squeeze(1) > 0.7, True, False)
                            v_labels_tf = np.where(vlabels.numpy() > 0.9, True, False)
                            bit_xor_1 = np.bitwise_not(np.bitwise_xor(vector_1, v_labels_tf))
                            bit_xor_2 = np.bitwise_not(np.bitwise_xor(vector_2, v_labels_tf))
                            bit_xor_3 = np.bitwise_not(np.bitwise_xor(vector_3, v_labels_tf))
                            bit_xor_4 = np.bitwise_not(np.bitwise_xor(vector_4, v_labels_tf))
                            bit_xor_5 = np.bitwise_not(np.bitwise_xor(vector_5, v_labels_tf))
                            running_true_accuracy_1 = running_true_accuracy + bit_xor_1.sum() / len(bit_xor_1)
                            running_true_accuracy_2 = running_true_accuracy + bit_xor_2.sum() / len(bit_xor_2)
                            running_true_accuracy_3 = running_true_accuracy + bit_xor_3.sum() / len(bit_xor_3)
                            running_true_accuracy_4 = running_true_accuracy + bit_xor_4.sum() / len(bit_xor_4)
                            running_true_accuracy_5 = running_true_accuracy + bit_xor_5.sum() / len(bit_xor_5)
                    self.net.train(True)  # Turn gradients back on for training
                    avg_loss = running_loss / 9
                    avg_vloss = running_vloss / len(self.test_loader)
                    avg_accuracy_prediction_1 = 100 * (running_true_accuracy_1 / len(self.test_loader))
                    avg_accuracy_prediction_2 = 100 * (running_true_accuracy_2 / len(self.test_loader))
                    avg_accuracy_prediction_3 = 100 * (running_true_accuracy_3 / len(self.test_loader))
                    avg_accuracy_prediction_4 = 100 * (running_true_accuracy_4 / len(self.test_loader))
                    avg_accuracy_prediction_5 = 100 * (running_true_accuracy_5 / len(self.test_loader))
                    # Log the running loss averaged per batch
                    self.writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)
                    self.writer.add_scalars('True accuracy', {'accuracy-0.5': avg_accuracy_prediction_1,
                                                         'accuracy-0.55': avg_accuracy_prediction_2,
                                                         'accuracy-0.6': avg_accuracy_prediction_3,
                                                         'accuracy-0.65': avg_accuracy_prediction_4,
                                                         'accuracy-0.7': avg_accuracy_prediction_5}, epoch + 1)
                    self.average_loss_validation.append(avg_vloss)
                    self.average_loss_training.append(avg_loss)
                    self.avg_accuracy_prediction_1.append(avg_accuracy_prediction_1)
                    self.avg_accuracy_prediction_2.append(avg_accuracy_prediction_2)
                    self.avg_accuracy_prediction_3.append(avg_accuracy_prediction_3)
                    self.avg_accuracy_prediction_4.append(avg_accuracy_prediction_4)
                    self.avg_accuracy_prediction_5.append(avg_accuracy_prediction_5)
                    self.epoch_number.append(epoch)
                    self.writer.flush()
                    running_loss = 0.0
            if avg_vloss < best_val_loss:  # Save the best model based on validation loss
                best_val_loss = avg_vloss
                torch.save(self.net.state_dict(), 'best_model_' + self.name + '_' + str(self.batch_size) + '.pt')
                counter = 0
            else:
                counter = counter + 1
            # Stop training if the validation loss hasn't improved for a certain number of epochs (patience)
            if counter >= patience:
                end_time = time.time()
                self.time_diff = end_time - start_time
                minutes, seconds = divmod(self.time_diff, 60)
                self.time_diff = f"{int(minutes)}m {int(seconds)}s"
                print("Early stopping at epoch {} model name {}".format(epoch, self.name))
                break
        torch.save(self.net.state_dict(), self.name + '_batch_size' + str(self.batch_size) + '.pt')
        print('Finished Training')
        self.writer.flush()

    def test_model(self):
        criterion = torch.nn.BCELoss()
        running_tloss, running_true_accuracy = 0.0, 0.0
        with torch.no_grad():
            self.net.train(False)  # Don't need to track gradients for validation
            for j, tdata in enumerate(self.test_loader, 0):
                tinputs, tlabels = tdata
                toutputs = self.net(tinputs)
                tloss = criterion(toutputs.squeeze(1), tlabels)
                running_tloss += tloss.item()
                vector_1 = np.where(toutputs.numpy().squeeze(1) > 0.5, True, False)
                vector_2 = np.where(toutputs.numpy().squeeze(1) > 0.55, True, False)
                vector_3 = np.where(toutputs.numpy().squeeze(1) > 0.6, True, False)
                vector_4 = np.where(toutputs.numpy().squeeze(1) > 0.65, True, False)
                vector_5 = np.where(toutputs.numpy().squeeze(1) > 0.7, True, False)
                v_labels_tf = np.where(tlabels.numpy() > 0.9, True, False)
                bit_xor_1 = np.bitwise_not(np.bitwise_xor(vector_1, v_labels_tf))
                bit_xor_2 = np.bitwise_not(np.bitwise_xor(vector_2, v_labels_tf))
                bit_xor_3 = np.bitwise_not(np.bitwise_xor(vector_3, v_labels_tf))
                bit_xor_4 = np.bitwise_not(np.bitwise_xor(vector_4, v_labels_tf))
                bit_xor_5 = np.bitwise_not(np.bitwise_xor(vector_5, v_labels_tf))
                running_true_accuracy_1 = running_true_accuracy + bit_xor_1.sum() / len(bit_xor_1)
                running_true_accuracy_2 = running_true_accuracy + bit_xor_2.sum() / len(bit_xor_2)
                running_true_accuracy_3 = running_true_accuracy + bit_xor_3.sum() / len(bit_xor_3)
                running_true_accuracy_4 = running_true_accuracy + bit_xor_4.sum() / len(bit_xor_4)
                running_true_accuracy_5 = running_true_accuracy + bit_xor_5.sum() / len(bit_xor_5)
            avg_tloss = running_tloss / len(self.test_loader)
            avg_accuracy_prediction_1 = 100 * (running_true_accuracy_1 / len(self.test_loader))
            avg_accuracy_prediction_2 = 100 * (running_true_accuracy_2 / len(self.test_loader))
            avg_accuracy_prediction_3 = 100 * (running_true_accuracy_3 / len(self.test_loader))
            avg_accuracy_prediction_4 = 100 * (running_true_accuracy_4 / len(self.test_loader))
            avg_accuracy_prediction_5 = 100 * (running_true_accuracy_5 / len(self.test_loader))
            # Log the running loss averaged per batch
            self.writer.add_scalars('Test set', {'Test loss': avg_tloss}, j + 1)
            self.writer.add_scalars('True accuracy Test set', {'accuracy-0.5': avg_accuracy_prediction_1,
                                                 'accuracy-0.55': avg_accuracy_prediction_2,
                                                 'accuracy-0.6': avg_accuracy_prediction_3,
                                                 'accuracy-0.65': avg_accuracy_prediction_4,
                                                 'accuracy-0.7': avg_accuracy_prediction_5}, j + 1)
        self.average_loss_test.append(avg_tloss)
        self.avg_accuracy_prediction_1_test.append(avg_accuracy_prediction_1)
        self.avg_accuracy_prediction_2_test.append(avg_accuracy_prediction_2)
        self.avg_accuracy_prediction_3_test.append(avg_accuracy_prediction_3)
        self.avg_accuracy_prediction_4_test.append(avg_accuracy_prediction_4)
        self.avg_accuracy_prediction_5_test.append(avg_accuracy_prediction_5)
        self.writer.flush()
        self.write_to_file()


