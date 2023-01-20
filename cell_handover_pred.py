import os
from collections import defaultdict
import folium as folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd
import branca.colormap as cm
import cell_calculation
from aux_functions import euclidean_distance, euclidean_distance_points, haversine, getDistanceBetweenPointsNew
from cell_calculation import get_unique_cells_in_drive
from regression_model import build_regression_model
import matplotlib.pyplot as mp

from visualization import prepare_distance_to_cells

# filter according to start of the drive. unique.
NUM_DRIVES = 1
DRIVE_NUM = 400


def init_dataset(pickle_name, drive_num):
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
            drives_by_imei_dictionary[key_for_dict] = sort_drives_by_modem[i][j]
    return sort_drives_by_modem, drives_by_imei_dictionary


# identifying the cells of each drive per modem. the key of the dict is when the drive started following by the
# modem_id. key=date_time_modemID
def get_cells_per_drive_in_dataset(drives_by_imei_dictionary):
    cells_per_drive_per_modem_avg = {}
    for key in drives_by_imei_dictionary.keys():  # len(drives_by_modem)
        cells, updated_df = get_unique_cells_in_drive(drives_by_imei_dictionary[key])
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


if __name__ == "__main__":
    drives_by_modem, returned_drives_by_imei_dict = init_dataset('pickle_rick.pkl', DRIVE_NUM)
    cells_per_drives_in_dataset, cells_dict = get_cells_per_drive_in_dataset(returned_drives_by_imei_dict)
    drives_by_imei_dict = prepare_switchover_col(returned_drives_by_imei_dict)
    x = 5
    # drives_by_imei_dict, rsrp_dictionary = prepare_distance_to_cells(drives_by_imei_dict, cells_dict)
    # visualize_drives(returned_drives_by_imei_dict, cells_per_drives_in_dataset)
    build_regression_model(drives_by_imei_dict)
