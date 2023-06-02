import warnings
import numpy as np
import pandas as pd
import torch
import pickle
import architecture
import input_parser
import training
from load_preprocess_ds import init_drives_dataset, get_cells_per_drive_in_dataset, prepare_switchover_col, \
    preprocess_features, training_sets_init
from training import prepare_data_sets
from torch.utils.data import TensorDataset, DataLoader, random_split

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parsed_args = input_parser.Parser()
    opts = parsed_args.parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opts.load_from_files == 0:
        returned_drives_by_imei_dict_train = init_drives_dataset('pickle_rick_full.pkl', opts.number_drives, opts.prepare_data_set)
        cells_per_drives_in_dataset_train, cells_dict_train = get_cells_per_drive_in_dataset(returned_drives_by_imei_dict_train)
        # visualization.visualize_drives(returned_drives_by_imei_dict_train, cells_per_drives_in_dataset_train)
        drives_by_imsi_dict = prepare_switchover_col(returned_drives_by_imei_dict_train)
        training_data = training_sets_init(drives_by_imsi_dict, opts.max_switch_over, opts.max_data_imsi)
        # correlated_data_dict_train = normalize_correlate_features(drives_by_imei_dict_train)
        correlated_data_dict_train = preprocess_features(training_data, label=opts.label)  # label = 1 = switchover, 0 = latency, 2 = loss_rate
        data_set_concat_train = pd.concat(correlated_data_dict_train, axis=0).reset_index()
        data_set_concat_train.drop(["level_0", "level_1"], axis=1, inplace=True)
        X_train_seq, y_train_label, x_val_seq, y_val_label = \
            prepare_data_sets(data_set_concat_train, SEQ_LEN=opts.sequence_length, balanced=opts.bdataset, name=opts.model_name, label=opts.label,
                              training=opts.prepare_data_set,nfeatures=opts.nfeatures)
        exit()
    else:  # load from saved data sets and train the model.
        if opts.label == 2:  # loss
            if opts.prepare_data_set == 1 and opts.to_train == 1:
                X_train_seq = training.make_Tensor(np.array(pickle.load(open('datasets-loss/x_train_' + opts.model_name + '.pkl', "rb"))))
                y_train_label = training.make_Tensor(np.array(pickle.load(open('datasets-loss/y_train_' + opts.model_name + '.pkl', "rb"))))
                x_val_seq = training.make_Tensor(np.array(pickle.load(open('datasets-loss/X_val_' + opts.model_name + '.pkl', "rb"))))
                y_val_label = training.make_Tensor(np.array(pickle.load(open('datasets-loss/y_val_' + opts.model_name + '.pkl', "rb"))))
        elif opts.label == 0 and opts.to_train == 1:  # latency
            if opts.prepare_data_set == 1:
                X_train_seq = training.make_Tensor(np.array(pickle.load(open('dataset-latency/' + str(opts.number_drives) + '-drives/x_train_' + opts.model_name + '.pkl', "rb"))))
                y_train_label = training.make_Tensor(np.array(pickle.load(open('dataset-latency/' + str(opts.number_drives) + '-drives/y_train_' + opts.model_name + '.pkl', "rb"))))
                x_val_seq = training.make_Tensor(np.array(pickle.load(open('dataset-latency/' + str(opts.number_drives) + '-drives/X_val_' + opts.model_name + '.pkl', "rb"))))
                y_val_label = training.make_Tensor(np.array(pickle.load(open('dataset-latency/' + str(opts.number_drives) + '-drives/y_val_' + opts.model_name + '.pkl', "rb"))))
        else:  # switch over label == 1
            if opts.prepare_data_set == 1 and opts.to_train == 1:
                X_train_seq = training.make_Tensor(np.array(pickle.load(open('training/x_' + opts.model_name + '.pkl', "rb"))))
                y_train_label = training.make_Tensor(np.array(pickle.load(open('training/y_train_' + opts.model_name + '.pkl', "rb"))))
                x_val_seq = training.make_Tensor(np.array(pickle.load(open('training/X_val_' + opts.model_name + '.pkl', "rb"))))
                y_val_label = training.make_Tensor(np.array(pickle.load(open('training/y_val_' + opts.model_name + '.pkl', "rb"))))
    if opts.to_train == 1:
        train_data_set = TensorDataset(X_train_seq, y_train_label)
        val_data_set = TensorDataset(x_val_seq, y_val_label)
        # Calculate split sizes
        # train_size = int(0.8 * len(train_data_set))
        # val_size = len(train_data_set) - train_size
        # train_dataset, val_dataset = random_split(train_data_set, [train_size, val_size])
        train_loader = DataLoader(train_data_set, batch_size=opts.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data_set, batch_size=opts.batch_size, shuffle=False, drop_last=True)
        features_count = X_train_seq.shape[2]
        training_class = training.optimizer(opts.model_name, opts.epoch_number, train_loader, val_loader, val_loader, opts.sequence_length,
                                            features_count, opts.neuralnetwork_size, opts.learn_rate, opts.batch_size, opts.label,device)
        if opts.label == 0 or opts.label == 2:
            training_class.main_training_loop_latency()
        else:
            training_class.main_training_loop_switchover()
        print("Finished training model " + opts.model_name + "_" + str(opts.batch_size))
    else:  # testing the model.
        if opts.nfeatures == 3:
            n_features = "allfeatures"
        elif opts.nfeatures == 2:
            n_features = "nogps"
        elif opts.nfeatures == 2:
            n_features = "rssr"
        else:
            n_features = "gps"
        x_test_seq = training.make_Tensor(np.array(pickle.load(open('dataset-latency/20-drives/x_seq_{}_{}_label_latency_{}_test.pkl'.format(
            opts.sequence_length, opts.number_drives, n_features), "rb"))))
        y_test_label = training.make_Tensor(np.array(pickle.load(open('dataset-latency/20-drives/y_seq_{}_{}_label_latency_{}_test.pkl'.format(
            opts.sequence_length, opts.number_drives, n_features), "rb"))))
        tensordataset = TensorDataset(x_test_seq, y_test_label)
        import math
        batch_size_x = math.floor(len(TensorDataset(x_test_seq, y_test_label)) / opts.batch_size)
        test_loader = DataLoader(tensordataset, batch_size=batch_size_x, shuffle=False, drop_last=True)
        model = architecture.cnn_lstm_hybrid(features=x_test_seq.shape[2], label=opts.label)
        model.load_state_dict(torch.load('best_model_{}_batch_size_{}.pt'.format(opts.model_name, opts.batch_size)))

        # x_test_seq1 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/only gps/x_seq_64_80_label_ho_only_lat_long_test'+ '.pkl', "rb"))))
        # y_test_label1 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/only gps/y_seq_64_80_label_ho_only_lat_long_test'+ '.pkl', "rb"))))
        # test_loader1 = DataLoader(TensorDataset(x_test_seq1, y_test_label1), batch_size=468, shuffle=False, drop_last=True)
        # model1 = architecture.cnn_lstm_hybrid(features=x_test_seq1.shape[2], label=opts.label)
        # model1.load_state_dict(torch.load('ROC for static 64/only gps/best_model_seq_64_80_label_ho_only_lat_long.pt'))
        #
        # x_test_seq2 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/rsrp/x_seq_64_80_label_ho_rsrp_test'+ '.pkl', "rb"))))
        # y_test_label2 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/rsrp/y_seq_64_80_label_ho_rsrp_test'+ '.pkl', "rb"))))
        # test_loader2 = DataLoader(TensorDataset(x_test_seq2, y_test_label2), batch_size=468, shuffle=False, drop_last=True)
        # model2 = architecture.cnn_lstm_hybrid(features=x_test_seq2.shape[2], label=opts.label)
        # model2.load_state_dict(torch.load('ROC for static 64/rsrp/best_model_seq_64_80_label_ho_rsrp'+ '.pt'))
        #
        # x_test_seq3 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/7 features/x_seq_64_80_label_ho_no_lat_long_test'+ '.pkl', "rb"))))
        # y_test_label3 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/7 features/y_seq_64_80_label_ho_no_lat_long_test'+ '.pkl', "rb"))))
        # test_loader3 = DataLoader(TensorDataset(x_test_seq3, y_test_label3), batch_size=468, shuffle=False, drop_last=True)
        # model3 = architecture.cnn_lstm_hybrid(features=x_test_seq3.shape[2], label=opts.label)
        # model3.load_state_dict(torch.load('ROC for static 64/7 features/best_model_seq_64_80_label_ho_no_lat_long'+ '.pt'))
        #
        # x_test_seq4 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/9 features/x_seq_64_80_label_ho_no_full_test' + '.pkl', "rb"))))
        # y_test_label4 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/9 features/y_seq_64_80_label_ho_no_full_test' + '.pkl', "rb"))))
        # test_loader4 = DataLoader(TensorDataset(x_test_seq4, y_test_label4), batch_size=468, shuffle=False, drop_last=True)
        # model4 = architecture.cnn_lstm_hybrid(features=x_test_seq4.shape[2], label=opts.label)
        # model4.load_state_dict(torch.load('ROC for static 64/9 features/best_model_seq_64_80_all_imsi_batch_size_512' + '.pt'))

        # x_test_seq1 = training.make_Tensor(np.array(pickle.load(open('ROC for 9 features-diff window/x_seq_32_80_label_ho_full_test'+ '.pkl', "rb"))))
        # y_test_label1 = training.make_Tensor(np.array(pickle.load(open('ROC for 9 features-diff window/y_seq_32_80_label_ho_full_test'+ '.pkl', "rb"))))
        # test_loader1 = DataLoader(TensorDataset(x_test_seq1, y_test_label1), batch_size=466, shuffle=False, drop_last=True)
        # model1 = architecture.cnn_lstm_hybrid(features=x_test_seq1.shape[2], label=opts.label)
        # model1.load_state_dict(torch.load('ROC for 9 features-diff window/best_model_seq_32_80_all_imsi_batch_size_512.pt'))
        #
        # x_test_seq2 = training.make_Tensor(np.array(pickle.load(open('ROC for 9 features-diff window/x_seq_64_80_label_ho_no_full_test'+ '.pkl', "rb"))))
        # y_test_label2 = training.make_Tensor(np.array(pickle.load(open('ROC for 9 features-diff window/y_seq_64_80_label_ho_no_full_test'+ '.pkl', "rb"))))
        # test_loader2 = DataLoader(TensorDataset(x_test_seq2, y_test_label2), batch_size=468, shuffle=False, drop_last=True)
        # model2 = architecture.cnn_lstm_hybrid(features=x_test_seq2.shape[2], label=opts.label)
        # model2.load_state_dict(torch.load('ROC for 9 features-diff window/best_model_seq_64_80_all_imsi_batch_size_512'+ '.pt'))
        #
        # x_test_seq3 = training.make_Tensor(np.array(pickle.load(open('ROC for 9 features-diff window/x_seq_128_80_label_ho_full_long_test'+ '.pkl', "rb"))))
        # y_test_label3 = training.make_Tensor(np.array(pickle.load(open('ROC for 9 features-diff window/y_seq_128_80_label_ho_full_long_test'+ '.pkl', "rb"))))
        # test_loader3 = DataLoader(TensorDataset(x_test_seq3, y_test_label3), batch_size=468, shuffle=False, drop_last=True)
        # model3 = architecture.cnn_lstm_hybrid(features=x_test_seq3.shape[2], label=opts.label)
        # model3.load_state_dict(torch.load('ROC for 9 features-diff window/best_model_seq_128_80_all_imsi_batch_size_512'+ '.pt'))
        #
        # x_test_seq4 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/9 features/x_seq_64_80_label_ho_no_full_test' + '.pkl', "rb"))))
        # y_test_label4 = training.make_Tensor(np.array(pickle.load(open('ROC for static 64/9 features/y_seq_64_80_label_ho_no_full_test' + '.pkl', "rb"))))
        # test_loader4 = DataLoader(TensorDataset(x_test_seq4, y_test_label4), batch_size=468, shuffle=False, drop_last=True)
        # model4 = architecture.cnn_lstm_hybrid(features=x_test_seq4.shape[2], label=opts.label)
        # model4.load_state_dict(torch.load('ROC for static 64/9 features/best_model_seq_64_80_all_imsi_batch_size_512' + '.pt'))
        if opts.label == 0 or opts.label == 2:
            training.test_model_mse(test_loader=test_loader, given_model=model, opts=opts)
        else:
            training.test_model_bce(test_loader1=test_loader1, given_model1=model1, test_loader2=test_loader2, given_model2=model2,
                                    test_loader3=test_loader3, given_model3=model3,
                                    test_loader4=test_loader4, given_model4=model4,opts=opts)
        print("Finished testing model " + opts.model_name + "_" + str(opts.batch_size))
        print("finished making a data set.")
