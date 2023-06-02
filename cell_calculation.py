import struct

import pandas as pd


def where_are_my_cells(cell_num):
    cell_location_israel = pd.read_csv('425_new.csv')
    cell_location_israel['cell'] = cell_location_israel['cell'].astype('str')
    mask = cell_location_israel['cell'].str.startswith(str(cell_num))
    result = cell_location_israel[mask].drop('radio', axis=1).astype(float)
    result = pd.DataFrame(result.mean()).transpose()  # average for cell
    if result.empty:
        mask = cell_location_israel['cell'].str.contains(str(cell_num))
        result = cell_location_israel[mask]
    return result


def where_are_my_cells_lat_long(cell_num):
    cell_location_israel = pd.read_csv('425_new.csv')
    cell_location_israel['cell'] = cell_location_israel['cell'].astype('str')
    mask = cell_location_israel['cell'].str.startswith(str(cell_num))
    result = cell_location_israel[mask].drop('radio', axis=1).astype(float)
    result = pd.DataFrame(result.mean()).transpose().dropna()  # average for cell
    if result.empty:
        mask = cell_location_israel['cell'].str.contains(str(cell_num))
        result = cell_location_israel[mask]
        return result
    return result['lat'].values[0], result['lon'].values[0]


# Define a custom function to remove ".0" from floats
def remove_zero(x):
    if isinstance(x, float) and str(x).endswith(".0"):
        return int(x)
    return x


def translate_hex(hex_num):
    int_num = int(hex_num, 16)  # Convert hex number to int
    int_num = int_num >> 8  # Remove lowest 2 digits
    return int(str(int_num), 10)


# calculate the effective cell by minimum distance to the cells.
def get_unique_cells_in_drive(df):
    df = df.applymap(remove_zero)
    df[['globalcellid']] = df[['globalcellid']].astype(str)  # .astype(int) # unique cells from the data-frame
    df.insert(18, "celldecimal", 0, allow_duplicates=False)
    df['celldecimal'] = df.apply(lambda row: translate_hex(row['globalcellid']), axis=1)
    unique_cells_global = df['globalcellid'].unique()
    unique_cells_decimal = df['celldecimal'].unique()
    cells_location_dict_global, cells_location_dict_decimal = {},{}
    for cell in unique_cells_global:
        if cell in cells_location_dict_global:
            continue
        else:
            cells_location_dict_global[cell] = where_are_my_cells_lat_long(cell)
            x = 5
    for cell in unique_cells_decimal:
        if cell in cells_location_dict_decimal:
            continue
        else:
            cells_location_dict_decimal[cell] = where_are_my_cells_lat_long(cell)

    # New dictionary to hold the processed values
    returned_dict_for_cells = {}

    for key, value in cells_location_dict_decimal.items():
        if isinstance(value, tuple):
            returned_dict_for_cells[key] = value
        elif isinstance(value, pd.DataFrame):
            returned_dict_for_cells[key] = (value['lat'].mean(), value['lon'].mean())
    # Remove entries with NaN values
    import numpy as np
    keys_to_remove = [key for key, value in returned_dict_for_cells.items() if np.isnan(value[0]) or np.isnan(value[1])]
    for key in keys_to_remove:
        del returned_dict_for_cells[key]
    return returned_dict_for_cells, df #cells_location_dict_global, cells_location_dict_decimal #  #, xxxxxxxxxxx


def calculate_switchover(df):
    old_global_cell = df.head(1).globalcellid.values[0]
    # old_decimal_cell = df.head(1).celldecimal.values[0]
    for idx, row in df.iterrows():
        if df.loc[idx, 'globalcellid'] == old_global_cell:
            continue
        if df.loc[idx, 'globalcellid'] != old_global_cell and df.loc[idx, 'globalcellid'] != 0:
            df.loc[idx, 'switchover_global'] = 1
            old_global_cell = df.loc[idx, 'globalcellid']

    # for idx, row in df.iterrows():
    #     if df.loc[idx, 'celldecimal'] == old_global_cell:
    #         continue
    #     if df.loc[idx, 'celldecimal'] != old_decimal_cell and df.loc[idx, 'celldecimal'] != 0:
    #         df.loc[idx, 'switchover_decimal'] = 1
    #         old_global_cell = df.loc[idx, 'celldecimal']
