import pandas as pd

def where_are_my_cells(cell_num):
    cell_location_israel = pd.read_csv('425_new.csv')
    cell_location_israel['cell'] = cell_location_israel['cell'].astype('str')
    mask = cell_location_israel['cell'].str.startswith(str(cell_num))
    result = cell_location_israel[mask]
    if result.empty:
        mask = cell_location_israel['cell'].str.contains(str(cell_num))
        result = cell_location_israel[mask]

    return result

def get_unique_cells_in_drive(df):
    unique_cells_in_this_drive = df['globalcellid'].unique()  # unique cells from the data-frame
    cell_dfs_list = []
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