import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

import cell_calculation
import seaborn as sns


def init_drives_dataset(pickle_name, starting_drive_number, number_of_drives_for_ds):
    big_df = pd.read_pickle(pickle_name)

    # for i in big_df.columns:
    #     # count number of rows with missing values
    #     n_miss = big_df[i].isnull().sum()
    #     perc = round(n_miss / big_df.shape[0] * 100, 3)
    #     print('col: {} is missing: {}'.format(i, perc))
    drives = [v for k, v in big_df.groupby(['date', 'time'])]

    sorted_drives = []
    for i in range(len(drives)):  # Now filter according to imei in each drive
        sorted_drives.append([v for k, v in drives[i].groupby('imsi')])
    # drive_by_modems is list of lists. each drive is a list of lists. the list inside is for diff imeis.

    # each drive is broken into imsi. There are 600 drives and at most 6 modems.
    drives_by_imsi_dictionary = {}
    for i in range(starting_drive_number, starting_drive_number + number_of_drives_for_ds):  # len(drives_by_modem)
        for j in range(len(sorted_drives[i])):  # len(drives_by_modem[i]) - number os modems in drive.
            key_for_dict = str(
                sorted_drives[i][j]['date'].iloc[0] + '_' + sorted_drives[i][j]['time'].iloc[0] + '_' +
                str(sorted_drives[i][j]['imsi'].iloc[0]))
            drives_by_imsi_dictionary[key_for_dict] = copy.copy(sorted_drives[i][j])
    return drives_by_imsi_dictionary


# identifying the cells of each drive per modem. the key of the dict is when the drive started following by the
# modem_id. key=date_time_modemID
def get_cells_per_drive_in_dataset(drives_by_imei_dictionary):
    cells_per_drive_per_modem_avg = {}
    for key in drives_by_imei_dictionary.keys():  # len(drives_by_modem)
        cells, updated_df = cell_calculation.get_unique_cells_in_drive(drives_by_imei_dictionary[key])
        # cells_per_drive_per_modem_avg[key] = pd.concat(cells, axis=0, join='outer').dropna(axis=0) @ TODO remove comments for cells visual
        drives_by_imei_dictionary[key] = updated_df
    cells_dictionary = {}
    # for key in cells_per_drive_per_modem_avg.keys():
    #     for index, row in cells_per_drive_per_modem_avg[key].iterrows():
    #         cells_dictionary[row['cell']] = (row['lon'], row['lat']) TODO remove comments for cells visual
    return cells_per_drive_per_modem_avg, cells_dictionary


def prepare_switchover_col(drives_by_imei_dictionary):
    for key in drives_by_imei_dictionary.keys():
        # drives_by_imei_dictionary[key]['celldecimal'] = drives_by_imei_dictionary[key]['celldecimal'].astype(float)
        drives_by_imei_dictionary[key] = drives_by_imei_dictionary[key].sort_values(by='timestamp')
        drives_by_imei_dictionary[key]['globalcellid'] = drives_by_imei_dictionary[key]['globalcellid'].astype(float)
        drives_by_imei_dictionary[key]['globalcellid'].replace(0, np.nan, inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].ffill(inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].replace(2013, np.nan, inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].ffill(inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].replace(1000, np.nan, inplace=True)
        drives_by_imei_dictionary[key]['globalcellid'].ffill(inplace=True)

        drives_by_imei_dictionary[key] = drives_by_imei_dictionary[key].reset_index().drop('index', axis=1)
        drives_by_imei_dictionary[key].insert(18, "switchover_global", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key].insert(18, "global_cell_id_shift", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key]['global_cell_id_shift'] = drives_by_imei_dictionary[key]['globalcellid'].shift()
        drives_by_imei_dictionary[key]['globalcellid'] = drives_by_imei_dictionary[key].apply(
            lambda y: y['global_cell_id_shift'] if y['globalcellid'] == 0 else y['globalcellid'], axis=1)
        drives_by_imei_dictionary[key].drop(["global_cell_id_shift"], axis=1, inplace=True)
        cell_calculation.calculate_switchover(drives_by_imei_dictionary[key])  # Calculate switch over per drive per imei
    return drives_by_imei_dictionary


def fill_missing_data_per_col(df, col_name):
    imputer = KNNImputer(n_neighbors=3)  # fill missing data
    latitude_col_imputed = imputer.fit_transform(df[[col_name]])
    df[col_name] = latitude_col_imputed
    return df


def preprocess_features(data_dict):
    # one_hot = pd.get_dummies(big_df['operator'], prefix='operator')  # make the operator into 1 hot encoding.
    # big_df = pd.concat([big_df, one_hot], axis=1)  # Concatenate the original DataFrame and the one-hot encoding
    labels_dict = {}
    for key in data_dict.keys():
        data_dict[key].drop(columns=['imei', 'drive_id'], inplace=True, axis=1)
        col_to_fill_missing_values = ['latitude', 'longitude', 'norm_bw', 'changes', 'end_state']
        for col in col_to_fill_missing_values:
            n_miss = data_dict[key][col].isnull().sum()
            perc = round(n_miss / data_dict[key].shape[0] * 100, 4)
            fill_missing_data_per_col(data_dict[key], col)  # fill the missing data per coloumn using KNNimputer with nearest neighbors.
            n_miss = data_dict[key][col].isnull().sum()
            perc = round(n_miss / data_dict[key].shape[0] * 100, 4)
            x = 5
        # TODO: should do here a correlation based .
        x = 6
        normalized_cols = ['longitude', 'latitude', 'rsrp', 'rssi', 'rsrq', 'modem_bandwidth', 'latency_mean',
                           'total_bitrate', 'frame_latency_mean', 'qp_mean', 'loss_rate', 'timestamp', 'switchover_global']
        scaler = MinMaxScaler()
        for col in normalized_cols:
            data_dict[key][col] = pd.DataFrame(scaler.fit_transform(data_dict[key][[col]]))

        labels_dict[key] = copy.copy(data_dict[key][normalized_cols])
        data_dict[key] = copy.copy(data_dict[key][normalized_cols])
        corr = data_dict[key].corr()
        # selected_columns = np.full((corr.shape[0],), True, dtype=bool)
        # for i in range(corr.shape[0]):
        #     for j in range(i + 1, corr.shape[0]):
        #         if corr.iloc[i, j] >= 0.9:
        #             if selected_columns[j]:
        #                 selected_columns[j] = False
        # selected_columns = data_dict[key].columns[selected_columns]
        # data_dict[key] = copy.copy(data_dict[key][selected_columns])

        # sns.heatmap(corr, fmt=".2f", linewidth=.2)
        # plt.title("Communication Test Drive Features")
        # plt.rcParams["figure.figsize"] = (30, 30)
        # # plt.xticks(fontsize=16)
        # plt.tight_layout()
        # plt.savefig("correlationmat.pdf", dpi=300)
        # x = 5
        # plt.show(dpi=300)
    return data_dict


def create_sequence(data_dict, seq_length):
    xs = []
    ys = []
    for i in range(len(data_dict) - seq_length):
        x = data_dict.drop(["switchover_global"], axis=1).iloc[i:(i + seq_length)]
        y = data_dict["switchover_global"].iloc[i + seq_length]
        x.dropna(inplace=True)
        if len(x) == seq_length:
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)


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
    else:  ##concatenate all of the drives with in the same IMSI, and then take the biggest one.
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
        # result_dict = results_return_key
        imei_key = sorted(result_dict, key=lambda k: len(result_dict[k]), reverse=True)[imsi_number]
        dict_to_return = {imei_key: copy.copy(result_dict[imei_key])}
        return dict_to_return
