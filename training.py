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
from sklearn.metrics import f1_score
import architecture
from load_preprocess_ds import create_sequence
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix


def make_Tensor(array):
    # return torch.Tensor(array).float()
    return torch.from_numpy(array).float()


def balance_data_set(seq, seq_label):
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


def prepare_data_sets(data_frame, SEQ_LEN, balanced, name, label):
    if balanced:  # under sampling
        so_idxs = data_frame.index[data_frame['switchover_global'] == 1].tolist()
        no_so_idxs = [random.randint(so_idxs[0], so_idxs[-1]) for _ in range(len(so_idxs)) if random.randint(so_idxs[0], so_idxs[-1]) not in so_idxs]
        # so_perc = 100 * (len(so_idxs) / len(data_frame))
        # so_perc = so_perc
        # nso = 100 - so_perc
        # sizes = [so_perc, nso]
        # fig, ax = plt.subplots()
        # labels = ['Switchover', "No Switchover"]
        # ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=200)
        # ax.axis('equal')
        # # Add a title to the chart
        # plt.title("Switchover Distribution")
        # plt.tight_layout()
        # plt.savefig("sodistribution.pdf", dpi=300)
        # plt.show()
        balanced_indexes = [j for i in [no_so_idxs, so_idxs] for j in i]
        random.shuffle(balanced_indexes)
        if label == 0:
            xs, ys = create_sequence(data_frame, SEQ_LEN, "latency_mean")
            xs = np.array(xs)
            ys = np.array(ys)
            data_set_size = xs.shape[0]
            train_size = int(data_set_size * 0.8)
            test_size = int(int(data_set_size - train_size) / 2)
            x_train, y_train = copy.copy(xs[:train_size]), copy.copy(ys[:train_size])
            X_val, y_val = copy.copy(xs[train_size:train_size + test_size]), copy.copy(ys[train_size:train_size + test_size])
            # X_test, y_test = copy.copy(xs[train_size + test_size:]), copy.copy(ys[train_size + test_size:])
        else:
            xs, ys = [], []
            for i in range(len(balanced_indexes)):
                x = data_frame.iloc[balanced_indexes[i] - SEQ_LEN - 1:balanced_indexes[i] - 1].drop(["switchover_global"],
                                                                                                    axis=1)  # that way it is to predict 1 second ahead.
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
            # X_test, y_test = copy.copy(xs[train_size + test_size:]), copy.copy(ys[train_size + test_size:])
    else:  # balance with SMOTE
        seq, seq_label = create_sequence(data_frame, SEQ_LEN)
        seq, seq_label = balance_data_set(seq, seq_label)  # over sampled
        data_set_size = seq.shape[0]
        train_size = int(data_set_size * 0.8)
        test_size = int(int(data_set_size - train_size))
        x_train, y_train = copy.copy(seq[:train_size]), copy.copy(seq_label[:train_size])
        X_val, y_val = copy.copy(seq[train_size:train_size + test_size]), copy.copy(
            seq_label[train_size:train_size + test_size])
    pickle.dump(x_train, open('x_train_test_' + name + '.pkl', "wb"))
    pickle.dump(y_train, open('y_train_test_' + name + '.pkl', "wb"))
    pickle.dump(X_val, open('X_val_' + name + '.pkl', "wb"))
    pickle.dump(y_val, open('y_val_' + name + '.pkl', "wb"))
    # pickle.dump(X_test, open('X_test_' + name + '.pkl', "wb"))
    # pickle.dump(y_test, open('y_test_' + name + '.pkl', "wb"))
    return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val)
    # return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
    #     make_Tensor(y_test)
    # return x_train, y_train, X_val, y_val, X_test, y_test


class optimizer:
    def __init__(self, name, epochs, training_loader, validation_loader, test_loader, seq_len, number_of_features, hidden_size, learning_rate,
                 batch_size, label):
        self.name = name
        self.train_loader = training_loader
        self.epochs = epochs
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.sequence_len = seq_len
        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.net = architecture.cnn_lstm_hybrid(features=number_of_features, label=label)
        self.learn_rate = learning_rate
        self.average_loss_validation, self.average_loss_training, self.average_loss_test = [], [], []
        self.avg_accuracy_prediction_1 = []
        self.avg_accuracy_prediction_3 = []
        self.avg_accuracy_prediction_5 = []
        self.avg_accuracy_prediction_1_test = []
        self.avg_accuracy_prediction_3_test = []
        self.avg_accuracy_prediction_5_test = []
        self.predicted_latency_values, self.real_latency_values = [], []
        self.epoch_number = []
        self.time_diff = 0
        self.batch_size = batch_size
        self.label = label
        self.writer = SummaryWriter('models/' + self.name + '_batch_size_' + str(self.batch_size))

    def write_to_file_switchover(self, flag):
        df_validation = pd.DataFrame({'avg_validation_loss': self.average_loss_validation,
                                      'avg_training_loss': self.average_loss_training,
                                      'accuracy_05_val': self.avg_accuracy_prediction_1,
                                      'accuracy_06_val': self.avg_accuracy_prediction_3,
                                      'accuracy_07_val': self.avg_accuracy_prediction_5,
                                      'epochs': self.epoch_number,
                                      'training time': self.time_diff,
                                      'tl_samples_batches_count': len(self.train_loader),
                                      'tl_sample_count': len(self.train_loader.dataset),
                                      'vl_samples_batches_count': len(self.validation_loader),
                                      'vl_sample_count': len(self.validation_loader.dataset),
                                      'testl_samples_batches_count': len(self.test_loader),
                                      'testl_sample_count': len(self.test_loader.dataset)})

        if flag is True:
            df_test = pd.DataFrame({
                'avg_test_loss': self.average_loss_test,
                'accuracy_05_test': self.avg_accuracy_prediction_1_test,
                'accuracy_06_test': self.avg_accuracy_prediction_3_test,
                'accuracy_07_test': self.avg_accuracy_prediction_5_test},
            )
            unified_df = pd.concat([df_validation, df_test], axis=0)
            unified_df.to_csv(
                'models/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_batch_size_' + str(self.batch_size) + '_' + str(
                    time.time()) + '.csv', index=False)
        else:
            df_validation.to_csv(
                'models/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_batch_size_' + str(self.batch_size) + '_' + str(
                    time.time()) + '.csv', index=False)

    def write_to_file_latency(self):
        df_validation = pd.DataFrame({'avg_validation_loss': self.average_loss_validation,
                                      'avg_training_loss': self.average_loss_training,
                                      'predictions': self.predicted_latency_values,
                                      'real_latency': self.real_latency_values,
                                      'epochs': self.epoch_number,
                                      'training time': self.time_diff,
                                      'tl_samples_batches_count': len(self.train_loader),
                                      'tl_sample_count': len(self.train_loader.dataset),
                                      'vl_samples_batches_count': len(self.validation_loader),
                                      'vl_sample_count': len(self.validation_loader.dataset)})
        df_validation.to_csv(
            'models/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_batch_size_' + str(self.batch_size) + '_' + str(
                time.time()) + '.csv', index=False)

    def main_training_loop_switchover(self):
        avg_vloss = float('inf')
        criterion = torch.nn.BCELoss()
        optim_to_learn = optim.Adam(self.net.parameters(), lr=self.learn_rate)
        best_val_loss = float('inf')
        counter = 0
        patience = 15
        # To view, start TensorBoard on the command line with:
        #   tensorboard --logdir=model/seq_64_20_all_imsi_batch_size_256
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
                    # print('Batch {}'.format(i + 1))
                    running_vloss, running_true_accuracy = 0.0, 0.0
                    with torch.no_grad():
                        self.net.train(False)  # Don't need to track gradients for validation
                        for j, vdata in enumerate(self.validation_loader, 0):
                            vinputs, vlabels = vdata
                            voutputs = self.net(vinputs)
                            vloss = criterion(voutputs.squeeze(1), vlabels)
                            running_vloss += vloss.item()
                            vector_1 = np.where(voutputs.numpy().squeeze(1) > 0.5, True, False)
                            vector_3 = np.where(voutputs.numpy().squeeze(1) > 0.6, True, False)
                            vector_5 = np.where(voutputs.numpy().squeeze(1) > 0.7, True, False)
                            v_labels_tf = np.where(vlabels.numpy() > 0.9, True, False)
                            bit_xor_1 = np.bitwise_not(np.bitwise_xor(vector_1, v_labels_tf))
                            bit_xor_3 = np.bitwise_not(np.bitwise_xor(vector_3, v_labels_tf))
                            bit_xor_5 = np.bitwise_not(np.bitwise_xor(vector_5, v_labels_tf))
                            running_true_accuracy_1 = running_true_accuracy + bit_xor_1.sum() / len(bit_xor_1)
                            running_true_accuracy_3 = running_true_accuracy + bit_xor_3.sum() / len(bit_xor_3)
                            running_true_accuracy_5 = running_true_accuracy + bit_xor_5.sum() / len(bit_xor_5)
                    self.net.train(True)  # Turn gradients back on for training
                    avg_loss = running_loss / 9
                    avg_vloss = running_vloss / len(self.validation_loader)
                    avg_accuracy_prediction_1 = 100 * (running_true_accuracy_1 / len(self.validation_loader))
                    avg_accuracy_prediction_3 = 100 * (running_true_accuracy_3 / len(self.validation_loader))
                    avg_accuracy_prediction_5 = 100 * (running_true_accuracy_5 / len(self.validation_loader))
                    # Log the running loss averaged per batch
                    self.writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)
                    self.writer.add_scalars('True accuracy', {'accuracy-0.5': avg_accuracy_prediction_1,
                                                              'accuracy-0.6': avg_accuracy_prediction_3,
                                                              'accuracy-0.7': avg_accuracy_prediction_5}, epoch + 1)
                    self.average_loss_validation.append(avg_vloss)
                    self.average_loss_training.append(avg_loss)
                    self.avg_accuracy_prediction_1.append(avg_accuracy_prediction_1)
                    self.avg_accuracy_prediction_3.append(avg_accuracy_prediction_3)
                    self.avg_accuracy_prediction_5.append(avg_accuracy_prediction_5)
                    self.epoch_number.append(epoch)
                    self.writer.flush()
                    running_loss = 0.0
            if avg_vloss < best_val_loss:  # Save the best model based on validation loss
                best_val_loss = avg_vloss
                torch.save(self.net.state_dict(),
                           'models/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + 'best_model_' + self.name + '_batch_size_' + str(
                               self.batch_size) + '.pt')
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
        torch.save(self.net.state_dict(), 'models/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_batch_size' +
                   str(self.batch_size) + str(time.time()) + '.pt')
        print('Finished Training')
        self.writer.flush()
        self.write_to_file_switchover(flag=False)

    def main_training_loop_latency(self):
        criterion = torch.nn.MSELoss()
        optim_to_learn = optim.Adam(self.net.parameters(), lr=self.learn_rate)
        best_val_loss = float('inf')
        counter = 0
        patience = 15
        # To view, start TensorBoard on the command line with:
        #   tensorboard --logdir=model/seq_64_20_all_imsi_batch_size_256
        # ...and open a browser tab to http://localhost:6006/
        start_time = time.time()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            train_loss = 0.0
            self.net.train()
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                optim_to_learn.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs.squeeze(1), labels)
                loss.backward()
                optim_to_learn.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            self.net.eval()
            val_loss = 0
            running_validation_predition = 0.0
            running_validation_labels = 0.0
            with torch.no_grad():
                for j, vdata in enumerate(self.validation_loader, 0):
                    vinputs, vlabels = vdata
                    voutputs = self.net(vinputs)
                    vloss = criterion(voutputs.squeeze(1), vlabels)
                    val_loss += vloss.item()
                    running_validation_predition += np.array(voutputs.squeeze(1).tolist()).mean()
                    running_validation_labels += np.array(vlabels.tolist()).mean()
            val_loss /= len(self.validation_loader)
            running_validation_predition /= len(self.validation_loader)
            running_validation_labels /= len(self.validation_loader)
            self.writer.add_scalars('Training vs. Validation Loss', {'Training': train_loss, 'Validation': val_loss}, epoch + 1)
            self.average_loss_training.append(train_loss)
            self.average_loss_validation.append(val_loss)
            self.predicted_latency_values.append(running_validation_predition)
            self.real_latency_values.append(running_validation_labels)
            if val_loss < best_val_loss:  # Save the best model based on validation loss
                best_val_loss = val_loss
                torch.save(self.net.state_dict(),
                           'models/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + 'best_model_' + self.name + '_batch_size_' + str(
                               self.batch_size) + '.pt')
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
        print('Finished Training')
        self.writer.flush()
        self.write_to_file_latency()


def test_model(test_loader, given_model, opts):
    net = given_model
    average_loss_test, true_positive_arr_06, false_pos_array_06 = [], [], []
    true_positive_arr_07, false_pos_array_07 = [], []
    writer = SummaryWriter('test/' + opts.model_name + '_batch_size_' + str(opts.batch_size))
    criterion = torch.nn.BCELoss()
    for i in range(0, 5):
        running_tloss, true_positive_running_06, false_positive_running_06 = 0.0, 0.0, 0.0
        true_positive_running_07, false_positive_running_07 = 0.0, 0.0
        with torch.no_grad():
            for j, tdata in enumerate(test_loader, 0):
                tinputs, tlabels = tdata
                toutputs = net(tinputs)
                tloss = criterion(toutputs.squeeze(1), tlabels)
                running_tloss += tloss.item()
                vector_3 = np.where(toutputs.numpy().squeeze(1) > 0.6, True, False)
                vector_5 = np.where(toutputs.numpy().squeeze(1) > 0.7, True, False)
                cm_06 = confusion_matrix(tlabels, vector_3, normalize='true')
                cm_07 = confusion_matrix(tlabels, vector_5, normalize='true')
                from sklearn import metrics
                from sklearn.metrics import ConfusionMatrixDisplay
                # cm_display_06 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_06, display_labels=[False, True])
                # cm_display_07 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_07, display_labels=[False, True])

                # Create the figure and subplots
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                # Plot each confusion matrix
                ConfusionMatrixDisplay(cm_06, display_labels=[False, True]).plot(ax=axs[0], cmap=plt.cm.Blues)
                ConfusionMatrixDisplay(cm_07, display_labels=[False, True]).plot(ax=axs[1], cmap=plt.cm.Blues)
                axs[0].set_title("Threshold 06")
                axs[1].set_title("Threshold 07")
                # plt.title("Sequence 64")
                plt.savefig("sequence-128-20.pdf", dpi=300)
                plt.show()

                true_positive_06 = cm_06[1][1]
                false_positive_06 = cm_06[0][1]
                true_positive_07 = cm_07[1][1]
                false_positive_07 = cm_07[0][1]
                true_positive_running_06 += true_positive_06
                false_positive_running_06 += false_positive_06
                true_positive_running_07 += true_positive_07
                false_positive_running_07 += false_positive_07
            avg_tloss = running_tloss / len(test_loader)
            avg_tpositive_06 = true_positive_running_06 / len(test_loader)
            avg_fpositive_06 = false_positive_running_06 / len(test_loader)
            avg_tpositive_07 = true_positive_running_07 / len(test_loader)
            avg_fpositive_07 = false_positive_running_07 / len(test_loader)
            average_loss_test.append(avg_tloss)
            true_positive_arr_06.append(true_positive_06)
            false_pos_array_06.append(false_positive_06)
            true_positive_arr_07.append(true_positive_07)
            false_pos_array_07.append(false_positive_07)
            # Log the running loss averaged per batch
            writer.add_scalars('Test set', {'Test loss': avg_tloss}, j + 1)
            writer.add_scalars('True accuracy Test set', {'true positive 06': avg_tpositive_06,
                                                          'false positive 06': avg_fpositive_06,
                                                          'true positive 07': avg_tpositive_07,
                                                          'false positive 07': avg_fpositive_07}, j + 1)

    writer.flush()
    write_test_to_file(true_positive=true_positive_arr_06, false_pos=false_pos_array_06, true_positive2=true_positive_arr_07,
                       false_pos2=false_pos_array_07, test_loss=average_loss_test, test_loader=test_loader, seq_len=opts.sequence_length)


def write_test_to_file(true_positive, false_pos, true_positive2, false_pos2, test_loss, test_loader, seq_len):
    df_validation = pd.DataFrame({'avg_test_loss': test_loss,
                                  'tp06': true_positive,
                                  'fp06': false_pos,
                                  'tp07': true_positive2,
                                  'fp07': false_pos2,
                                  'testl_samples_batches_count': len(test_loader),
                                  'testl_sample_count': len(test_loader.dataset)})
    df_validation.to_csv(
        'test/' + str(seq_len) + str(time.time()) + '.csv', index=False)
