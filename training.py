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
import architecture
from load_preprocess_ds import create_sequence
from sklearn.metrics import confusion_matrix


def make_Tensor(array):
    # return torch.Tensor(array).float()
    return torch.from_numpy(array).float()


# def balance_data_set(seq, seq_label):
#     sm = SMOTE(sampling_strategy='minority', random_state=42)
#     seq_data_reshaped_for_balancing = seq.reshape(seq.shape[0], -1)
#     oversampled_seq, oversampled_labels = sm.fit_resample(seq_data_reshaped_for_balancing, seq_label)
#     oversampled_seq = oversampled_seq.reshape(oversampled_seq.shape[0], seq.shape[1], seq.shape[2])
#     return oversampled_seq, oversampled_labels


def prepare_data_sets(data_frame, SEQ_LEN, balanced, name, label, training,nfeatures):
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
        if label == 0 or label == 2:
            label_text = "latency_mean" if label == 0 else "loss_rate"
            xs, ys = create_sequence(data_frame, SEQ_LEN, label_text,nfeatures)
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
                x = data_frame.iloc[balanced_indexes[i] - SEQ_LEN - 1:balanced_indexes[i] - 1].drop(["switchover_global","latency_mean"],axis=1)  # that way it is to predict 1 second ahead.
                y = data_frame["switchover_global","latency_mean"].iloc[balanced_indexes[i]]
                x.dropna(inplace=True)
                if len(x) == SEQ_LEN:
                    xs.append(x)
                    ys.append(y)
            xs = np.array(xs)
            ys = np.array(ys)
            data_set_size = xs.shape[0]
            train_size = int(data_set_size * 0.8)
            test_size = int(int(data_set_size - train_size))
            x_train, y_train = copy.copy(xs[:train_size]), copy.copy(ys[:train_size])
            X_val, y_val = copy.copy(xs[train_size:train_size + test_size]), copy.copy(ys[train_size:train_size + test_size])
            # X_test, y_test = copy.copy(xs[train_size + test_size:]), copy.copy(ys[train_size + test_size:])
    if training == 1:
        pickle.dump(x_train, open('x_train_' + name + '.pkl', "wb"))
        pickle.dump(y_train, open('y_train_' + name + '.pkl', "wb"))
        pickle.dump(X_val, open('X_val_' + name + '.pkl', "wb"))
        pickle.dump(y_val, open('y_val_' + name + '.pkl', "wb"))
    else:
        pickle.dump(x_train, open('x_' + name + '.pkl', "wb"))
        pickle.dump(y_train, open('y_' + name + '.pkl', "wb"))
    exit()
    # pickle.dump(X_test, open('X_test_' + name + '.pkl', "wb"))
    # pickle.dump(y_test, open('y_test_' + name + '.pkl', "wb"))
    # return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val)
    # return make_Tensor(x_train), make_Tensor(y_train), make_Tensor(X_val), make_Tensor(y_val), make_Tensor(X_test), \
    #     make_Tensor(y_test)
    # return x_train, y_train, X_val, y_val, X_test, y_test


class optimizer:
    def __init__(self, name, epochs, training_loader, validation_loader, test_loader, seq_len, number_of_features, hidden_size, learning_rate,
                 batch_size, label,device):
        self.name = name
        self.train_loader = training_loader
        self.epochs = epochs
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.sequence_len = seq_len
        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.net = architecture.cnn_lstm_hybrid(features=number_of_features, label=label)
        self.device = device
        self.net.to(device)
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
        self.writer = SummaryWriter('new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(self.batch_size))

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
                'new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_batch_size_' + str(
                    self.batch_size) + '_' + str(
                    time.time()) + '.csv', index=False)
        else:
            df_validation.to_csv(
                'new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_batch_size_' + str(
                    self.batch_size) + '_' + str(
                    time.time()) + '.csv', index=False)

    def write_to_file_loss(self):
        df_validation = pd.DataFrame({'avg_validation_loss': self.average_loss_validation,
                                      'avg_training_loss': self.average_loss_training,
                                      'predictions': self.predicted_latency_values,
                                      'real_latency': self.real_latency_values,
                                      'tl_samples_batches_count': len(self.train_loader),
                                      'tl_sample_count': len(self.train_loader.dataset),
                                      'vl_samples_batches_count': len(self.validation_loader),
                                      'vl_sample_count': len(self.validation_loader.dataset)})
        df_validation.to_csv(
            'new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + self.name + '_' + str(
                time.time()) + '.csv', index=False)

    def main_training_loop_switchover(self):
        avg_vloss = float('inf')
        criterion = torch.nn.BCELoss()
        optim_to_learn = optim.Adam(self.net.parameters(), lr=self.learn_rate)
        best_val_loss = float('inf')
        counter = 0
        patience = 20
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
                           'new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(self.batch_size) + "/" + 'best_model_' + self.name + '_batch_size_' + str(
                               self.batch_size) + '_' + str(time.time()) + '.pt')
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
        torch.save(self.net.state_dict(),
                   'new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(
                       self.batch_size) + "/" + 'best_model_' + self.name + '_batch_size_' + str(
                       self.batch_size) + '_' + str(time.time()) + '.pt')
        print('Finished Training')
        self.writer.flush()
        self.write_to_file_switchover(flag=False)

    def main_training_loop_latency(self):
        criterion = torch.nn.L1Loss()
        optim_to_learn = optim.Adam(self.net.parameters(), lr=self.learn_rate)
        best_val_loss = float('inf')
        counter = 0
        patience = 30
        start_time = time.time()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            train_loss = 0.0
            self.net.train()
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
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
                    vinputs, vlabels = vdata[0].to(self.device), vdata[1].to(self.device)
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
                           'new_runs_models_label_' + str(self.label) + '/' + self.name + '_batch_size_' + str(
                               self.batch_size) + "/" + 'best_model_' + self.name + '_batch_size_' + str(
                               self.batch_size) + '_' + str(time.time()) + '.pt')
                self.write_to_file_loss()
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
        print("Finished training model name:{}".format(self.name))
        self.writer.flush()
        self.write_to_file_loss()

def test_model_bce(test_loader1, given_model1, test_loader2, given_model2,test_loader3, given_model3
                   ,test_loader4, given_model4,opts):
    net1,net2,net3,net4 = given_model1, given_model2, given_model3, given_model4
    writer = SummaryWriter('test1/' + opts.model_name + '_batch_size_' + str(opts.batch_size))
    criterion = torch.nn.BCELoss()
    for i in range(0, 1):
        with torch.no_grad():
            for j, tdata in enumerate(test_loader1, 0):
                tinputs1, tlabels1 = tdata
                toutputs1 = net1(tinputs1)

        with torch.no_grad():
            for j, tdata in enumerate(test_loader2, 0):
                tinputs2, tlabels2 = tdata
                toutputs2 = net2(tinputs2)

        with torch.no_grad():
            for j, tdata in enumerate(test_loader3, 0):
                tinputs3, tlabels3 = tdata
                toutputs3 = net3(tinputs3)
        with torch.no_grad():
            for j, tdata in enumerate(test_loader4, 0):
                tinputs4, tlabels4 = tdata
                toutputs4 = net4(tinputs4)

        # vector_3 = np.where(toutputs3.numpy().squeeze(1) > 0.6, True, False)
        # vector_5 = np.where(toutputs3.numpy().squeeze(1) > 0.7, True, False)
        # cm_06 = confusion_matrix(tlabels3, vector_3, normalize='true')
        # cm_07 = confusion_matrix(tlabels3, vector_5, normalize='true')
        # from sklearn.metrics import ConfusionMatrixDisplay
        # fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # ConfusionMatrixDisplay(cm_06, display_labels=[False, True]).plot(ax=axs[0], cmap=plt.cm.Blues)
        # ConfusionMatrixDisplay(cm_07, display_labels=[False, True]).plot(ax=axs[1], cmap=plt.cm.Blues)
        # axs[0].set_title("Threshold 06")
        # axs[1].set_title("Threshold 07")
        # # plt.title("Sequence 64")
        # plt.savefig("9-features-128"+ ".pdf", dpi=300)
        # plt.show()
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        # Define the predictions and labels for each model
        preds1, labels1 = toutputs1, tlabels1
        preds2, labels2 = toutputs2, tlabels2
        preds3, labels3 = toutputs3, tlabels3
        preds4, labels4 = toutputs4, tlabels4
        # Calculate the false positive rate and true positive rate for each model
        fpr1, tpr1, _ = roc_curve(labels1, preds1)
        fpr2, tpr2, _ = roc_curve(labels2, preds2)
        fpr3, tpr3, _ = roc_curve(labels3, preds3)
        fpr4, tpr4, _ = roc_curve(labels4, preds4)

        # Calculate the area under the curve for each model
        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)
        roc_auc4 = auc(fpr4, tpr4)
        plt.figure(figsize=(12, 10))
        # plt.plot(fpr1, tpr1, color='blue', lw=2, label='32 window length (AUC = %0.2f)' % roc_auc1)
        # plt.plot(fpr2, tpr2, color='red', lw=2, label='64 window length (AUC = %0.2f)' % roc_auc2)
        # plt.plot(fpr3, tpr3, color='orange', lw=2, label='128 window length (AUC = %0.2f)' % roc_auc3)
        # plt.plot(fpr3, tpr3, color='green', lw=1, label='9 Features (AUC = %0.2f)' % roc_auc4)

        plt.plot(fpr1, tpr1, color='blue', lw=2, label='GPS only (AUC = %0.2f)' % roc_auc1)
        plt.plot(fpr2, tpr2, color='red', lw=2, label='RSRP & RSRQ (AUC = %0.2f)' % roc_auc2)
        plt.plot(fpr4, tpr4, color='orange', lw=2, label='7 Features (AUC = %0.2f)' % roc_auc4)
        plt.plot(fpr3, tpr3, color='green', lw=2, label='9 Features (AUC = %0.2f)' % roc_auc3)

        # # Plot the ROC curves for all three models on the same plot
        # plt.plot(fpr1, tpr1, color='blue', lw=1, label='GPS (AUC = %0.2f)' % roc_auc1)
        # plt.plot(fpr2, tpr2, color='red', lw=1, label='RSRP and RSRQ (AUC = %0.2f)' % roc_auc2)
        # # plt.plot(fpr4, tpr4, color='orange', lw=1, label='7 Features (AUC = %0.2f)' % roc_auc4)
        # plt.plot(fpr3, tpr3, color='green', lw=1, label='9 Features (AUC = %0.2f)' % roc_auc3)
        # # Plot the random model curve
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

        # Set the x and y limits of the plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # Add a grid to the plot
        plt.grid(True)
        # Add labels and legends to the plot
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc="lower right", fontsize=16)

        # Increase the size of the tick labels
        plt.tick_params(axis='both', which='major', labelsize=16)
        # plt.title('ROC Curves for Four different input features for a sliding window of 64 seconds')
        # plt.title('Different input features')
        # Save the plot
        plt.savefig("diff-features-64-static"+".pdf", dpi=600, bbox_inches='tight')
        # Display the plot
        plt.tight_layout()
        plt.show()
        x =5
    # write_test_to_file_bce(true_positive=true_positive_arr_06, false_pos=false_pos_array_06, true_positive2=true_positive_arr_07,
    #                        false_pos2=false_pos_array_07, test_loss=average_loss_test, test_loader=test_loader, seq_len=opts.sequence_length)

# def test_model_bce(test_loader, given_model, opts):
#     net = given_model
#     average_loss_test, true_positive_arr_06, false_pos_array_06 = [], [], []
#     true_positive_arr_07, false_pos_array_07 = [], []
#     writer = SummaryWriter('test/' + opts.model_name + '_batch_size_' + str(opts.batch_size))
#     criterion = torch.nn.BCELoss()
#     for i in range(0, 1):
#         running_tloss, true_positive_running_06, false_positive_running_06 = 0.0, 0.0, 0.0
#         true_positive_running_07, false_positive_running_07 = 0.0, 0.0
#         with torch.no_grad():
#             for j, tdata in enumerate(test_loader, 0):
#                 tinputs, tlabels = tdata
#                 toutputs = net(tinputs)
#                 tloss = criterion(toutputs.squeeze(1), tlabels)
#                 running_tloss += tloss.item()
#                 vector_3 = np.where(toutputs.numpy().squeeze(1) > 0.6, True, False)
#                 vector_5 = np.where(toutputs.numpy().squeeze(1) > 0.7, True, False)
#                 cm_06 = confusion_matrix(tlabels, vector_3, normalize='true')
#                 cm_07 = confusion_matrix(tlabels, vector_5, normalize='true')
#                 from sklearn import metrics
#                 from sklearn.metrics import ConfusionMatrixDisplay
#                 # cm_display_06 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_06, display_labels=[False, True])
#                 # cm_display_07 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_07, display_labels=[False, True])
#
#                 # Create the figure and subplots
#                 fig, axs = plt.subplots(1, 2, figsize=(12, 4))
#                 # Plot each confusion matrix
#                 ConfusionMatrixDisplay(cm_06, display_labels=[False, True]).plot(ax=axs[0], cmap=plt.cm.Blues)
#                 ConfusionMatrixDisplay(cm_07, display_labels=[False, True]).plot(ax=axs[1], cmap=plt.cm.Blues)
#                 axs[0].set_title("Threshold 06")
#                 axs[1].set_title("Threshold 07")
#                 # plt.title("Sequence 64")
#                 plt.savefig("no-gps-sequence"+ opts.model_name+ ".pdf", dpi=300)
#                 plt.show()
#
#             #     true_positive_06 = cm_06[1][1]
#             #     false_positive_06 = cm_06[0][1]
#             #     true_positive_07 = cm_07[1][1]
#             #     false_positive_07 = cm_07[0][1]
#             #     true_positive_running_06 += true_positive_06
#             #     false_positive_running_06 += false_positive_06
#             #     true_positive_running_07 += true_positive_07
#             #     false_positive_running_07 += false_positive_07
#             # avg_tloss = running_tloss / len(test_loader)
#             # avg_tpositive_06 = true_positive_running_06 / len(test_loader)
#             # avg_fpositive_06 = false_positive_running_06 / len(test_loader)
#             # avg_tpositive_07 = true_positive_running_07 / len(test_loader)
#             # avg_fpositive_07 = false_positive_running_07 / len(test_loader)
#             # average_loss_test.append(avg_tloss)
#             # true_positive_arr_06.append(true_positive_06)
#             # false_pos_array_06.append(false_positive_06)
#             # true_positive_arr_07.append(true_positive_07)
#             # false_pos_array_07.append(false_positive_07)
#             # # Log the running loss averaged per batch
#             # writer.add_scalars('Test set', {'Test loss': avg_tloss}, j + 1)
#             # writer.add_scalars('True accuracy Test set', {'true positive 06': avg_tpositive_06,
#             #                                               'false positive 06': avg_fpositive_06,
#             #                                               'true positive 07': avg_tpositive_07,
#             #                                               'false positive 07': avg_fpositive_07}, j + 1)
#
#     # writer.flush()
#     # write_test_to_file_bce(true_positive=true_positive_arr_06, false_pos=false_pos_array_06, true_positive2=true_positive_arr_07,
#     #                        false_pos2=false_pos_array_07, test_loss=average_loss_test, test_loader=test_loader, seq_len=opts.sequence_length)


def test_model_mse(test_loader, given_model, opts):
    net = given_model
    average_loss_test = []
    predictions, true_labels, ratio = [], [], []
    # writer = SummaryWriter('test/' + opts.model_name + '_batch_size_' + str(opts.batch_size))
    criterion = torch.nn.L1Loss()
    for i in range(0, 1):
        running_tloss = 0.0
        running_test_predition, running_test_labels = 0.0, 0.0
        with torch.no_grad():
            for j, tdata in enumerate(test_loader, 0):
                tinputs, tlabels = tdata
                toutputs = net(tinputs)
                tloss = criterion(toutputs.squeeze(1), tlabels)
                # print('model:{} and the MAE loss is: {}'.format(opts.model_name, tloss.item()))
                running_tloss += tloss.item()
                running_test_predition += np.array(toutputs.squeeze(1).tolist()).mean()
                running_test_labels += np.array(tlabels.tolist()).mean()
            avg_tloss = running_tloss / len(test_loader)
            print('model:{} and the MAE loss is: {}'.format(opts.model_name, avg_tloss))
            running_test_predition /= len(test_loader)
            running_test_labels /= len(test_loader)
            average_loss_test.append(avg_tloss)
            predictions.append(running_test_predition)
            true_labels.append(running_test_labels)
            ratio.append(running_test_predition / running_test_labels)
    write_test_to_file_mse(test_loss=average_loss_test, pred_arr=predictions, label_arr=true_labels, ratio_arr=ratio, seq_len=opts.sequence_length,
                           test_loader=test_loader)


def write_test_to_file_bce(true_positive, false_pos, true_positive2, false_pos2, test_loss, test_loader, seq_len):
    df_validation = pd.DataFrame({'avg_test_loss': test_loss,
                                  'tp06': true_positive,
                                  'fp06': false_pos,
                                  'tp07': true_positive2,
                                  'fp07': false_pos2,
                                  'testl_samples_batches_count': len(test_loader),
                                  'testl_sample_count': len(test_loader.dataset)})
    df_validation.to_csv(
        'test/' + str(seq_len) + str(time.time()) + '.csv', index=False)


def write_test_to_file_mse(test_loss, pred_arr, label_arr, ratio_arr, seq_len, test_loader):
    df_validation = pd.DataFrame({'avg_test_loss': test_loss,
                                  'pred': pred_arr,
                                  'true-label': label_arr,
                                  'ratio': ratio_arr,
                                  'testl_samples_batches_count': len(test_loader),
                                  'testl_sample_count': len(test_loader.dataset)})
    df_validation.to_csv('test/' + str(seq_len) + str(time.time()) + '.csv', index=False)
