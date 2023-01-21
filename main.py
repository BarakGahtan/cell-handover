import copy
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
from data_loader import init_dataset, get_cells_per_drive_in_dataset, prepare_switchover_col, \
    normalize_correlate_features
from regression_model import build_regression_model
import matplotlib.pyplot as mp
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
from visualization import prepare_distance_to_cells

# filter according to start of the drive. unique.
NUM_DRIVES = 1
DRIVE_NUM = 400

if __name__ == "__main__":
    drives_by_modem, returned_drives_by_imei_dict = init_dataset('pickle_rick.pkl', DRIVE_NUM)
    cells_per_drives_in_dataset, cells_dict = get_cells_per_drive_in_dataset(returned_drives_by_imei_dict)
    drives_by_imei_dict = prepare_switchover_col(returned_drives_by_imei_dict)
    correlated_data_dict = normalize_correlate_features(drives_by_imei_dict)

    # drives_by_imei_dict, rsrp_dictionary = prepare_distance_to_cells(drives_by_imei_dict, cells_dict)
    # visualize_drives(returned_drives_by_imei_dict, cells_per_drives_in_dataset)
    build_regression_model(drives_by_imei_dict)
