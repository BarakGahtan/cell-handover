import os

import folium as folium
from folium.plugins import HeatMap, MarkerCluster
from cell_reader import where_are_my_cells
import numpy as np
import pandas as pd

def read_log(file_name):
    path = file_name
    df = pd.read_csv(path)
    return df

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(row):
    x1 = row['latitude_perimeter']
    y1 = row['longitude_perimeter']
    x2 = row['latitude_perimeter_shift']
    y2 = row['longitude_perimeter_shift']
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


file_name = '20221212_154546_ironhide_measurements_internal.csv'
drive_df = read_log(file_name)
unique_cells_in_this_drive = drive_df['globalcellid'].unique()
cell_dfs_list = []
for cell in unique_cells_in_this_drive:
    cell_name = int('0x' + cell[:-2], 16)
    cells_locations = where_are_my_cells(cell_name)
    if cells_locations.empty:
        cells_locations = where_are_my_cells(cell[:-2])
    if cells_locations.empty:
        print('oops')
    cell_dfs_list.append(cells_locations)

all_cells_df = pd.concat(cell_dfs_list, axis=0, join='outer')


# Create a map object
locations = drive_df[['latitude_perimeter', 'longitude_perimeter']]
locationlist = locations.values.tolist()
m = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
unique_IMEIs = drive_df['imei'].unique()
modems_holder = {}
stat_holder = {}
map_speed = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
map_switchover = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
map_loss = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
map_rsrp = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
for index, row in all_cells_df.iterrows():
    folium.Circle((row['lat'], row['lon']), radius=row['range']/2, fill=True, ocacity=0.9, weight=2).add_to(map_switchover)

for imei in unique_IMEIs:
    modems_holder[imei] = drive_df[drive_df['imei'] == imei]
    modems_holder[imei].replace(999999, np.nan, inplace=True)
    modems_holder[imei].ffill(inplace=True)
    modems_holder[imei].bfill(inplace=True)
    modems_holder[imei]['globacellid_shift'] = modems_holder[imei]['globalcellid'].shift()
    modems_holder[imei]['switchover'] = modems_holder[imei].apply(lambda x: 0 if x['globalcellid'] == x['globacellid_shift'] else 1, axis=1)
    modems_holder[imei]['latitude_perimeter_shift'] = modems_holder[imei]['latitude_perimeter'].shift()
    modems_holder[imei]['longitude_perimeter_shift'] = modems_holder[imei]['longitude_perimeter'].shift()

    # Calculate the Euclidean distance between the two points for each row
    modems_holder[imei]['distance'] = modems_holder[imei].apply(euclidean_distance, axis=1)
    modems_holder[imei]['speed'] = modems_holder[imei]['distance'] * 60 * 60 * 111 * -1
    modems_holder[imei].fillna(0, inplace=True)
    # Add a layer with markers
    # data1 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'speed']].groupby(['latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
    data1 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'speed']]
    data_filt1 = data1.values.tolist()
    HeatMap(data=data_filt1, use_local_extrema=False, min_opacity=0.5,
            max_opacity=0.9,
            radius=25).add_to(folium.FeatureGroup(name=str(imei) + ' speed').add_to(map_speed))

    # data2 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'switchover']].groupby(['latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
    data2 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'switchover']]
    data2 = data2[data2['switchover'] == 1]
    marker_cluster = MarkerCluster(name=str(imei) + ' cluster').add_to(map_switchover)
    for index, row in data2.iterrows():
        folium.Marker(location=[row['latitude_perimeter'], row['longitude_perimeter']], popup=row['switchover']).add_to(marker_cluster)

    data_filt2 = data2.values.tolist()
    gradient = {0.1: 'green', 0.3: 'red'}
    HeatMap(data=data_filt2, use_local_extrema=False, min_opacity=0.0, gradient=gradient,
            max_opacity=0.95,
            radius=10).add_to(folium.FeatureGroup(name=str(imei) + ' switchover').add_to(map_switchover))

    # data3 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'loss_rate']].groupby(['latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
    data3 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'loss_rate']]
    data3 = data3[data3['loss_rate'] > 0]

    marker_cluster_loss = MarkerCluster(name=str(imei) + ' cluster').add_to(map_loss)
    for index, row in data3.iterrows():
        folium.Marker(location=[row['latitude_perimeter'], row['longitude_perimeter']], popup=row['loss_rate']).add_to(marker_cluster_loss)
    data_filt3 = data3.values.tolist()
    HeatMap(data=data_filt3, use_local_extrema=False, min_opacity=0.5,
            max_opacity=0.9,
            radius=25).add_to(folium.FeatureGroup(name=str(imei) + ' loss').add_to(map_loss))

    # data4 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'rsrp']].groupby(['latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
    data4 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'rsrp']]
    data_filt4 = data4.values.tolist()
    HeatMap(data=data_filt4, use_local_extrema=False, min_opacity=0.5,
            max_opacity=0.9,
            radius=25).add_to(folium.FeatureGroup(name=str(imei) + ' rsrp').add_to(map_rsrp))

    corr_mat = modems_holder[imei].corr()
    corr_mat.to_csv(str(imei) + '_corr_mat.csv')
    stat_holder[imei] = modems_holder[imei].describe()[['switchover', 'loss_rate', 'rsrp', 'latency_mean']].loc['mean']
    # corr_mat = modems_holder[imei].describe()
    # corr_mat.to_csv(str(imei) + '_stats.csv')

folium.LayerControl().add_to(map_speed)
map_speed.save('speed.html')
folium.LayerControl().add_to(map_switchover)
map_switchover.save('switchover.html')
folium.LayerControl().add_to(map_loss)
map_loss.save('loss.html')
folium.LayerControl().add_to(map_rsrp)
map_rsrp.save('rsrp.html')
pd.DataFrame.from_dict(stat_holder).to_csv('stats_compare.csv')

print(drive_df)
