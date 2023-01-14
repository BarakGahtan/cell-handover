import pandas as pd


def where_are_my_cells(cell_num):
    cell_location_israel = pd.read_csv('425_new.csv')
    cell_location_israel['cell'] = cell_location_israel['cell'].astype('str')
    mask = cell_location_israel['cell'].str.startswith(str(cell_num))
    result = cell_location_israel[mask].drop('radio', axis=1).astype(float)
    # result = pd.DataFrame(result.mean()).transpose()  # average for cell
    if result.empty:
        mask = cell_location_israel['cell'].str.contains(str(cell_num))
        result = cell_location_israel[mask]

    return result


def get_unique_cells_in_drive(df):
    unique_cells_in_this_drive = df['globalcellid'].unique()  # unique cells from the data-frame
    cell_dfs_list = []
    cell_name = ''
    for cell in unique_cells_in_this_drive:
        if isinstance(cell, float):
            if cell == 0:
                cell = '0'
                cell_name = int('0')
            else:
                cell = str(int(cell))
                cell_name = int('0x' + cell[:-2], 16)
        cells_locations = where_are_my_cells(cell_name)
        if cells_locations.empty:
            cells_locations = where_are_my_cells(cell[:-2])
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