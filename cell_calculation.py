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


# Define a custom function to remove ".0" from floats
def remove_zero(x):
    if isinstance(x, float) and str(x).endswith(".0"):
        return int(x)
    return x


def translate_hex(hex_num):
    int_num = int(hex_num, 16)  # Convert hex number to int
    int_num = int_num >> 8  # Remove lowest 2 digits
    return int(str(int_num), 10)


#caculate the effective cell by minimum distance to the cells.
def get_unique_cells_in_drive(df):
    df = df.applymap(remove_zero)
    df[['globalcellid']] = df[['globalcellid']].astype(str)  # .astype(int) # unique cells from the data-frame
    df.insert(18, "celldecimal", 0, allow_duplicates=False)
    df['celldecimal'] = df.apply(lambda row: translate_hex(row['globalcellid']), axis=1)
    unique_cells = pd.concat([df['globalcellid'], df['celldecimal']]).unique()
    x = 5
    cell_dfs_list = []
    for cell in unique_cells:
        cells_locations = where_are_my_cells(cell)
        if cells_locations.empty:
            print('oops')
        cell_dfs_list.append(cells_locations)
    return cell_dfs_list


def calculate_switchover(df):
    old_global_cell = df.head(1).globalcellid.values[0]
    current_global_cel = -1
    for idx, row in df.iterrows():
        if df.loc[idx, 'globalcellid'] == old_global_cell:
            continue
        if df.loc[idx, 'globalcellid'] != old_global_cell and df.loc[idx, 'globalcellid'] != 0:
            df.loc[idx, 'switchover'] = 1
            old_global_cell = df.loc[idx, 'globalcellid']
