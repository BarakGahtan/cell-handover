import copy
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import cell_calculation

def init_drives_dataset(pickle_name, number_of_drives_for_ds,testing):
    # with open(pickle_name, 'rb') as file:
    #     obj = pickle.Unpickler(file).load()
    big_df = pd.read_pickle(pickle_name)
    if testing == 0:
        big_df = big_df[big_df['date'] >= '20230208'] #testing
    else:
        big_df = big_df[big_df['date'] >= '20221201']
        big_df = big_df[big_df['date'] < '20230208' ]
        x =6
    # big_df.drop(columns=['imei', 'changes', 'end_state', 'operator', 'drive_id', 'rssi', 'latency_max', 'qp_mean', 'frame_lost',
    #                      'frame_latency_mean', 'latency_mean'], inplace=True, axis=1) #with latency_mean
    # big_df.drop(columns=['imei', 'imsi', 'changes', 'end_state', 'operator', 'drive_id','globalcellid'], axis=1)  # with latency_mean
    # # plt.figure(figsize=(15, 10))
    # cor = big_df.corr().abs()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Red, fmt=".2f")
    # plt.xticks(rotation=0)  # Set X-axis labels rotation to 0 degrees (horizontal)
    # plt.yticks(rotation=0)  # Set Y-axis labels rotation to 0 degrees (horizontal)
    # plt.tight_layout()
    # plt.savefig("corrmatrixfull-newrides.pdf", dpi=300,bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    # upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
    # to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]
    # big_df = big_df[big_df['date'] > '20221201']
    for i in big_df.columns:
        # count number of rows with missing values
        n_miss = big_df[i].isnull().sum()
        perc = round(n_miss / big_df.shape[0] * 100, 3)
        print('col: {} is missing: {}'.format(i, perc))
    drives = [v for k, v in big_df.groupby(['date', 'time'])]
    sorted_drives = []
    for i in range(len(drives)):  # Now filter according to imei in each drive
        sorted_drives.append([v for k, v in drives[i].groupby('imsi')])
    # drive_by_modems is list of lists. each drive is a list of lists. the list inside is for diff imeis.

    # each drive is broken into imsi. There are 600 drives and at most 6 modems.
    drives_by_imsi_dictionary = {}
    count = number_of_drives_for_ds
    counter = 0
    for i in range(len(sorted_drives) - 1, -1, -1):  # len(drives_by_modem)
        if counter == count:
            break
        for j in range(len(sorted_drives[i])):  # len(drives_by_modem[i]) - number of imsis in drive.
            key_for_dict = str(
                sorted_drives[i][j]['date'].iloc[0] + '_' + sorted_drives[i][j]['time'].iloc[0] + '_' +
                str(sorted_drives[i][j]['imsi'].iloc[0]))
            drives_by_imsi_dictionary[key_for_dict] = copy.copy(sorted_drives[i][j])
            # drives_by_imsi_dictionary[key_for_dict] = copy.copy(sorted_drives[i][j].drop(columns=to_drop, axis=1))
        counter = counter + 1
    return drives_by_imsi_dictionary


# identifying the cells of each drive per modem. the key of the dict is when the drive started following by the
# modem_id. key=date_time_modemID
def get_cells_per_drive_in_dataset(drives_by_imei_dictionary):
    cells_per_drive_per_modem_avg = {}
    for key in drives_by_imei_dictionary.keys():  # len(drives_by_modem)
        cells, updated_df = cell_calculation.get_unique_cells_in_drive(drives_by_imei_dictionary[key])
        df1 = pd.DataFrame(list(cells.items()), columns=['Key', 'Values']).apply(lambda row: pd.Series([row[0], *row[1]], index=['cell', 'lat', 'lon']), axis=1)
        # Check if there is a previous dataframe and concatenate, otherwise just assign df1
        if key in cells_per_drive_per_modem_avg:
            cells_per_drive_per_modem_avg[key] = pd.concat([cells_per_drive_per_modem_avg[key], df1], axis=0).dropna(axis=0)
        else:
            cells_per_drive_per_modem_avg[key] = df1
        drives_by_imei_dictionary[key] = updated_df
    cells_dictionary = {}
    for key in cells_per_drive_per_modem_avg.keys():
        for index, row in cells_per_drive_per_modem_avg[key].iterrows():
            cells_dictionary[row['cell']] = (row['lat'], row['lon']) #TODO remove comments for cells visual
    return cells_per_drive_per_modem_avg, cells_dictionary


def prepare_switchover_col(drives_by_imei_dictionary):
    for key in drives_by_imei_dictionary.keys():
        # drives_by_imei_dictionary[key]['celldecimal'] = drives_by_imei_dictionary[key]['celldecimal'].astype(float)
        drives_by_imei_dictionary[key] = drives_by_imei_dictionary[key].sort_values(by='timestamp')
        # drives_by_imei_dictionary[key]['globalcellid'] = drives_by_imei_dictionary[key]['globalcellid']
        drives_by_imei_dictionary[key]['globalcellid'].replace(0, np.nan, inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].ffill(inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].replace(2013, np.nan, inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].ffill(inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].replace(1000, np.nan, inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].ffill(inplace=True)

        drives_by_imei_dictionary[key] = drives_by_imei_dictionary[key].reset_index().drop('index', axis=1)
        drives_by_imei_dictionary[key].insert(8, "switchover_global", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key].insert(8, "global_cell_id_shift", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key]['global_cell_id_shift'] = drives_by_imei_dictionary[key]['globalcellid'].shift()
        drives_by_imei_dictionary[key]['globalcellid'] = drives_by_imei_dictionary[key].apply(
            lambda y: y['global_cell_id_shift'] if y['globalcellid'] == 0 else y['globalcellid'], axis=1)
        drives_by_imei_dictionary[key].drop(["global_cell_id_shift"], axis=1, inplace=True)
        cell_calculation.calculate_switchover(drives_by_imei_dictionary[key])  # Calculate switch over per drive per imei
    return drives_by_imei_dictionary


def fill_missing_data_per_col(df, col_name):
    imputer = KNNImputer(n_neighbors=2)  # fill missing data
    latitude_col_imputed = imputer.fit_transform(df[[col_name]])
    df[col_name] = latitude_col_imputed
    return df


def preprocess_features(data_dict, label):  # label is 1 if switchover, 0 if latency
    for key in data_dict.keys():
        # data_dict[key].drop(columns=['imei', 'drive_id', 'changes', 'end_state'], inplace=True, axis=1)
        col_to_fill_missing_values = data_dict[key].columns.tolist()
        for col in col_to_fill_missing_values:
            n_miss = data_dict[key][col].isnull().sum()
            perc = round(n_miss / data_dict[key].shape[0] * 100, 4)
            if perc != 0:
                fill_missing_data_per_col(data_dict[key], col)  # fill the missing data per coloumn using KNNimputer with nearest neighbors.
        data_dict[key].drop(["date", "time", "imsi", "globalcellid","operator"], axis=1, inplace=True)
        normalized_cols = data_dict[key].columns.tolist()
        scaler = MinMaxScaler()
        if label == 1:
            for col in normalized_cols:
                if col == 'switchover_global':
                    continue
                data_dict[key][col] = pd.DataFrame(scaler.fit_transform(data_dict[key][[col]]))
        elif label == 2:
            for col in normalized_cols:
                data_dict[key][col] = pd.DataFrame(scaler.fit_transform(data_dict[key][[col]]))
        else:
            for col in normalized_cols:
                data_dict[key][col] = pd.DataFrame(scaler.fit_transform(data_dict[key][[col]]))
    return data_dict


def create_sequence(data_dict, seq_length, label,nfeatures):
    xs = []
    ys = []
    for i in range(len(data_dict) - seq_length - 1):
        # feautures used: Timestamp, lat, lon, rsrp, rsrq, modem_bw, norm_bw, frame_lost, total_bitrate, loss_rate - 10 features
        if nfeatures == 3:
            x = data_dict.drop([label,"celldecimal","imei","switchover_global","frame_latency_mean","drive_id","end_state",
                            "latency_max","rssi","changes","qp_mean"], axis=1).iloc[i:(i + seq_length)] #all features except the label
        elif nfeatures == 2:
        # # feautures used: Timestamp,rsrp, rsrq, modem_bw, norm_bw, frame_lost, total_bitrate, loss_rate - 8 features
            x = data_dict.drop([label, "celldecimal", "imei","switchover_global", "drive_id","frame_latency_mean", "end_state",
                                "latency_max", "rssi", "changes", "qp_mean", "latitude", "longitude"], axis=1).iloc[
                i:(i + seq_length)]  # all features except the label
        #
        # feautures used:  rsrp, rsrq
        elif nfeatures == 1:
            x = data_dict.drop([label,"modem_bandwidth", "norm_bw", "frame_lost", "frame_latency_mean", "total_bitrate", "loss_rate" ,"switchover_global"
                                   ,"timestamp", "latitude", "longitude","celldecimal", "imei", "drive_id", "end_state", "latency_max", "rssi", "changes", "qp_mean"]
                               , axis=1).iloc[i:(i + seq_length)]
        else:
        # # feautures used:  "latitude", "longitude",
            x = data_dict.drop([label,"modem_bandwidth", "norm_bw", "frame_lost", "frame_latency_mean", "total_bitrate", "loss_rate" ,"switchover_global"
                                   ,"timestamp", "rsrp", "rsrq","celldecimal", "imei", "drive_id", "end_state", "latency_max", "rssi", "changes", "qp_mean"]
                               , axis=1).iloc[i:(i + seq_length)]

        y = data_dict[label].iloc[i + seq_length + 1]  # predict one second ahead
        x.dropna(inplace=True)
        if len(x) == seq_length:
            xs.append(x)
            ys.append(y)
    return xs, ys


def training_sets_init(given_dict, max_switchover, imsi_number):
    result_dict = {}
    if max_switchover == 1:
        keys = given_dict.keys()
        special_character = "_"
        keys_drives = set([s[:s.index(special_character, s.index(special_character) + 1)] for s in keys])
        for key in keys_drives:
            count_so, max_count = 0, 0
            for k in given_dict.keys():
                if k.startswith(key):
                    count_so = given_dict[k]['switchover_global'].eq(1).sum()
                    if count_so >= max_count:
                        result_dict[key] = copy.copy(given_dict[k])
                        max_count = count_so
                    else:
                        continue
        return result_dict
    else:  ## concatenate all of the drives with in the same IMSI, and then take the biggest one.
        keys = given_dict.keys()
        special_character = "_"
        keys_imsi = set([key.split(special_character)[-1] for key in keys if special_character in key])
        for key in keys_imsi:
            for k in given_dict.keys():
                if k.endswith(key):
                    if key in result_dict:
                        result_dict[key] = pd.concat([result_dict[key], copy.copy(given_dict[k])], ignore_index=True)
                    else:
                        result_dict[key] = copy.copy(given_dict[k])
        # highest_imei_key = max(result_dict, key=lambda x: len(result_dict[x]))
        # results_return_key = {highest_imei_key: copy.copy(result_dict[highest_imei_key])}
        # # result_dict = results_return_key
        # imei_key = sorted(result_dict, key=lambda k: len(result_dict[k]), reverse=True)[imsi_number]
        # dict_to_return = {imei_key: copy.copy(result_dict[imei_key])}
        return result_dict
