import warnings

import pandas as pd
import torch.optim as optim
import input_parser
import architecture
from architecture import cnn1d_model
from load_drives import init_drives_dataset, get_cells_per_drive_in_dataset, prepare_switchover_col, \
    normalize_correlate_features, training_sets_init
from training import prepare_data_sets, train_model
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# filter according to start of the drive. unique.


if __name__ == "__main__":
    parsed_args = input_parser.Parser()
    opts = parsed_args.parse()
    NUM_DRIVES = opts.number_drives
    DRIVE_NUM_TRAIN = opts.starting_drive_train
    DRIVE_NUM_TEST = opts.starting_drive_test
    SEQ_LEN = opts.sequence_length
    NN_SIZE = opts.neuralnetwork_size
    NN_LAYERS = opts.neuralnetwork_layers
    LSTM_FLAG = opts.lstm_enable
    BALANCED_FLAG = opts.bdataset
    CNN_FLAG = opts.cnn_enable
    drives_by_imei_train, returned_drives_by_imei_dict_train, list_of_drives = init_drives_dataset('pickle_rick.pkl', DRIVE_NUM_TRAIN, NUM_DRIVES)
    cells_per_drives_in_dataset_train, cells_dict_train = get_cells_per_drive_in_dataset(returned_drives_by_imei_dict_train)
    drives_by_imei_dict_train = prepare_switchover_col(returned_drives_by_imei_dict_train)
    training_data_by_so = training_sets_init(drives_by_imei_dict_train)
    # correlated_data_dict_train = normalize_correlate_features(drives_by_imei_dict_train)
    correlated_data_dict_train = normalize_correlate_features(training_data_by_so)
    data_set_concat_train = pd.concat(correlated_data_dict_train, axis=0).reset_index()
    data_set_concat_train.drop(["level_0", "level_1"], axis=1, inplace=True)  # should go into 1D-CNN MODEL
    X_train_seq, y_train_label, x_val_seq, y_val_label, x_test_seq, y_test_label = prepare_data_sets(data_set_concat_train, SEQ_LEN=SEQ_LEN,
                                                                                                     balanced=True)
    train_data_set = TensorDataset(X_train_seq,y_train_label)
    train_loader = DataLoader(train_data_set,batch_size=opts.batch_size,shuffle=False, drop_last=True)
    val_data_set = TensorDataset(x_val_seq,y_val_label)
    val_loader = DataLoader(val_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    test_data_set = TensorDataset(x_test_seq,y_test_label)
    test_loader = DataLoader(test_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    # DATA IS TENSORS

    input_dim = len(X_train_seq.shape[2])
    output_dim = 1
    hidden_dim = NN_SIZE
    layer_dim = NN_LAYERS
    batch_size = opts.batch_size
    dropout = 0.2
    n_epochs = opts.epoch_number
    learning_rate = 1e-3
    weight_decay = 1e-6

    model_params = {'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'layer_dim': layer_dim,
                    'output_dim': output_dim,
                    'dropout_prob': dropout}

    model = get_model('lstm', model_params)

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
    opt.plot_losses()

    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=input_dim)


















    cnn_model = cnn1d_model(seq_len=SEQ_LEN, number_of_features=X_train_seq.shape[2])  # number features is the seqeunce len * max pooling of Conv1D
    combined_model = architecture.cnn_lstm_combined(cnn_model, number_features=X_train_seq.shape[2],
                                                      n_hidden=NN_LAYERS, seq_len=SEQ_LEN,
                                                      n_layers=NN_LAYERS, cnn_enable=0)  # seq_len - delta t window to look back.
    model, train_hist, val_hist = train_model(combined_model, X_train_seq, y_train_label, val_data=x_val_seq,
                                              val_labels=y_val_label, lstm_flag=LSTM_FLAG)
    architecture.plot_train(train_hist, val_hist)

    x = 5
    # learning_model.test_model(x_test_seq, y_test_label, model)

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
