import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cell_calculation
import seaborn as sns


def init_drives_dataset(pickle_name, DRIVE_NUM, NUM_DRIVES):
    big_df = pd.read_pickle(pickle_name)
    drives = [v for k, v in big_df.groupby(['date', 'time'])]
    # Now filter according to modem in each drive
    sort_drives_by_modem = []
    for i in range(len(drives)):
        sort_drives_by_modem.append([v for k, v in drives[i].groupby('imei')])
    # drive_by_modems is list of lists. each drive is a list of lists. the list inside is for diff modems.

    # each drive is broken into imei. There are 600 drives and at most 6 modems.
    drives_by_imei_dictionary = {}
    for i in range(DRIVE_NUM, DRIVE_NUM + NUM_DRIVES):  # len(drives_by_modem)
        for j in range(len(sort_drives_by_modem[i])):  # len(drives_by_modem[i]) - number os modems in drive.
            key_for_dict = str(
                sort_drives_by_modem[i][j]['date'].iloc[0] + '_' + sort_drives_by_modem[i][j]['time'].iloc[
                    0] + '_' +
                str(sort_drives_by_modem[i][j]['imei'].iloc[0]))
            drives_by_imei_dictionary[key_for_dict] = copy.copy(sort_drives_by_modem[i][j])
    return sort_drives_by_modem, drives_by_imei_dictionary


# identifying the cells of each drive per modem. the key of the dict is when the drive started following by the
# modem_id. key=date_time_modemID
def get_cells_per_drive_in_dataset(drives_by_imei_dictionary):
    cells_per_drive_per_modem_avg = {}
    for key in drives_by_imei_dictionary.keys():  # len(drives_by_modem)
        cells, updated_df = cell_calculation.get_unique_cells_in_drive(drives_by_imei_dictionary[key])
        cells_per_drive_per_modem_avg[key] = pd.concat(cells, axis=0, join='outer').dropna(axis=0)
        drives_by_imei_dictionary[key] = updated_df
    cells_dictionary = {}
    for key in cells_per_drive_per_modem_avg.keys():
        for index, row in cells_per_drive_per_modem_avg[key].iterrows():
            cells_dictionary[row['cell']] = (row['lon'], row['lat'])
    return cells_per_drive_per_modem_avg, cells_dictionary


def prepare_switchover_col(drives_by_imei_dictionary):
    for key in drives_by_imei_dictionary.keys():
        drives_by_imei_dictionary[key]['celldecimal'] = drives_by_imei_dictionary[key]['celldecimal'].astype(float)
        drives_by_imei_dictionary[key]['globalcellid'] = drives_by_imei_dictionary[key]['globalcellid'].astype(float)
        drives_by_imei_dictionary[key].reset_index(inplace=True)
        drives_by_imei_dictionary[key].insert(18, "switchover_decimal", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key].insert(18, "decimal_cell_shift", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key]['decimal_cell_shift'] = drives_by_imei_dictionary[key]['celldecimal'].shift()
        drives_by_imei_dictionary[key]['celldecimal'] = drives_by_imei_dictionary[key].apply(
            lambda x: x['decimal_cell_shift'] if x['celldecimal'] == 0 else x['celldecimal'], axis=1)
        drives_by_imei_dictionary[key].drop(["decimal_cell_shift"], axis=1, inplace=True)

        drives_by_imei_dictionary[key].insert(18, "switchover_global", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key].insert(18, "global_cell_id_shift", 0, allow_duplicates=False)
        drives_by_imei_dictionary[key]['global_cell_id_shift'] = drives_by_imei_dictionary[key]['globalcellid'].shift()
        drives_by_imei_dictionary[key]['globalcellid'] = drives_by_imei_dictionary[key].apply(
            lambda y: y['global_cell_id_shift'] if y['globalcellid'] == 0 else y['globalcellid'], axis=1)
        drives_by_imei_dictionary[key].drop(["global_cell_id_shift"], axis=1, inplace=True)
        cell_calculation.calculate_switchover(
            drives_by_imei_dictionary[key])  # Calculate switch over per drive per imei
    return drives_by_imei_dictionary


def normalize_correlate_features(data_dict):
    labels_dict = {}
    for key in data_dict.keys():
        data_dict[key].drop(
            columns=['_id', 'date', 'time',
                     'latency_quantile_95', 'band', 'changes', 'longitude_perimeter', 'latitude_perimeter',
                     'client_id', 'modem_id', 'network_type', 'operator', 'index', 'latency_max', 'latency_median',
                     'latency_min', 'positionPrecision', 'servingcellid', 'qp_quantile_90', 'qp_min', 'qp_median',
                     'simIdentifier', 'source_name', 'end_state', 'distance_meters', 'norm_bw',
                     'frame_latency_quantile_90', 'frame_latency_min', 'frame_latency_median'], inplace=True, axis=1)
        normalized_cols = ['longitude', 'latitude', 'rsrp', 'rssi', 'rsrq', 'modem_bandwidth', 'latency_mean',
                           'total_bitrate', 'frame_latency_mean', 'timestamp', 'qp_mean']
        data_dict[key][normalized_cols] = (data_dict[key][normalized_cols] - data_dict[key][normalized_cols].min()) / (
                data_dict[key][normalized_cols].max() - data_dict[key][normalized_cols].min())
        data_cols = ['longitude', 'latitude', 'rsrp', 'rssi', 'rsrq', 'modem_bandwidth', 'latency_mean',
                     'total_bitrate', 'frame_latency_mean', 'timestamp', 'globalcellid', 'celldecimal', 'loss_rate',
                     'qp_mean', 'switchover_global']
        labels_dict[key] = copy.copy(data_dict[key][data_cols])
        data_dict[key] = copy.copy(data_dict[key][data_cols])
        corr = data_dict[key].corr()
        selected_columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.9:
                    if selected_columns[j]:
                        selected_columns[j] = False
        selected_columns = data_dict[key].columns[selected_columns]
        data_dict[key] = copy.copy(data_dict[key][selected_columns])
        sns.heatmap(corr, fmt=".2f", linewidth=.2)
        plt.title("Communication Test Drive Features")
        plt.rcParams["figure.figsize"] = (30, 30)
        # plt.xticks(fontsize=16)
        plt.tight_layout()
        # plt.savefig("correlationmat.pdf", dpi=300)
        x = 5
        # plt.show(dpi=300)
    return data_dict


def create_seq(data_dict, seq_length):
    xs = []
    ys = []
    for i in range(len(data_dict) - seq_length):
        x = data_dict.iloc[i:(i + seq_length)]
        y = data_dict.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)