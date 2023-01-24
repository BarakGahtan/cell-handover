import warnings
import pandas as pd
from matplotlib import pyplot as plt

import learning_model
from learning_model import cnn_extractor
from load_drives import init_drives_dataset, get_cells_per_drive_in_dataset, prepare_switchover_col, \
    normalize_correlate_features
from regression_model import build_regression_model
from training_model import prepare_data_sets, make_Tensor, train_model

warnings.filterwarnings("ignore")

# filter according to start of the drive. unique.
NUM_DRIVES = 15
DRIVE_NUM_TRAIN = 300
DRIVE_NUM_TEST = 500
SEQ_LEN = 10

if __name__ == "__main__":
    drives_by_modem_train, returned_drives_by_imei_dict_train = init_drives_dataset('pickle_rick.pkl', DRIVE_NUM_TRAIN,
                                                                                    NUM_DRIVES)
    cells_per_drives_in_dataset_train, cells_dict_train = get_cells_per_drive_in_dataset(
        returned_drives_by_imei_dict_train)
    drives_by_imei_dict_train = prepare_switchover_col(returned_drives_by_imei_dict_train)
    correlated_data_dict_train = normalize_correlate_features(drives_by_imei_dict_train)
    data_set_concat_train = pd.concat(correlated_data_dict_train, axis=0).reset_index()
    data_set_concat_train.drop(["level_0", "level_1"], axis=1, inplace=True)  # should go into 1D-CNN MODEL
    X_train_seq, y_train_label, x_val_seq, y_val_label, x_test_seq, y_test_label = prepare_data_sets(data_set_concat_train, SEQ_LEN=SEQ_LEN)
    # DATA IS TENSORS
    cnn_model = cnn_extractor()   #number features is the seqeunce len * max pooling of Conv1D
    combined_model = learning_model.cnn_lstm_combined(cnn_model, number_features=1600,
                                                      n_hidden=3, seq_len=SEQ_LEN,
                                                      n_layers=3)  # seq_len - delta t window to look back.
    model, train_hist, val_hist = train_model(combined_model, X_train_seq, y_train_label, val_data=x_val_seq, val_labels=y_val_label)
    learning_model.plot_train(train_hist, val_hist)
    learning_model.test_model(x_test_seq, y_test_label, model)

    # SET TEST DATA
    # drives_by_modem_test, returned_drives_by_imei_dict_test = init_drives_dataset('pickle_rick.pkl', DRIVE_NUM_TEST,
    #                                                                               NUM_DRIVES)
    # cells_per_drives_in_dataset_test, cells_dict_test = get_cells_per_drive_in_dataset(
    #     returned_drives_by_imei_dict_test)
    # drives_by_imei_dict_test = prepare_switchover_col(returned_drives_by_imei_dict_test)
    # correlated_data_dict_test = normalize_correlate_features(drives_by_imei_dict_test)
    # data_set_concat_test = pd.concat(correlated_data_dict_test, axis=0).reset_index()
    # data_set_concat_test = data_set_concat_test.drop(["level_0", "level_1"], axis=1,
    #                                                  inplace=True)  # should go into 1D-CNN MODEL
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # drives_by_imei_dict, rsrp_dictionary = prepare_distance_to_cells(drives_by_imei_dict, cells_dict)
    # visualize_drives(returned_drives_by_imei_dict_train, cells_per_drives_in_dataset_train)
    # build_regression_model(drives_by_imei_dict_train)
