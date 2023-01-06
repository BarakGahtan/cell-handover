import os
from collections import defaultdict
import folium as folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd
import branca.colormap

from cell_calculation import get_unique_cells_in_drive

picke_name = 'pickle_rick.pkl'
big_df = pd.read_pickle(picke_name)
drives = [v for k, v in big_df.groupby(['date', 'time'])]
#Now filter according to modem in each drive
drives_by_modem = []
for i in range(len(drives)):
    drives_by_modem.append([v for k, v in drives[i].groupby('modem_id')])
#each drive is broken into 6 modems. There are 600 drives and at most 6 modems.

cells_per_drive_per_modem_avg = {}
for i in range(1):  # len(drives_by_modem)
    y = 5
    for j in range(len(drives_by_modem[i])):
        key_to_insert = str(drives_by_modem[i][j]['date'].iloc[0] + '_' + drives_by_modem[i][j]['time'].iloc[0] + '_' +
                            drives_by_modem[i][j]['modem_id'].iloc[0])
        cells_per_drive_per_modem_avg[key_to_insert] = pd.concat(get_unique_cells_in_drive(drives_by_modem[i][j]), axis=0,
                                                                 join='outer').dropna(axis=0)
x = 5


for key, value in cells_per_drive_per_modem_avg.items():
    locations_in_drive_per_modem = cells_per_drive_per_modem_avg[key][['lat', 'lon']]
    locations_to_list = locations_in_drive_per_modem.values.toList()
    x = 5

x=5
# Create a map object
locations = drive_df[['latitude_perimeter', 'longitude_perimeter']]
locationlist = locations.values.tolist()
map_object = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
unique_IMEIs = drive_df['imei'].unique()
modems_holder = {}
stat_holder = {}
map_speed = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
map_switchover = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
map_loss = folium.Map(location=[32.17752, 34.93073], zoom_start=15)
map_rsrp = folium.Map(location=[32.17752, 34.93073], zoom_start=15)

#making the cells according to range
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
    import branca.colormap as cm

    colormap = cm.linear.RdYlGn_11.scale(0, 35).to_step(20)
    colormap.caption = 'Green is greater velocity'
    map_speed.add_child(colormap)

    data1 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'speed']]
    data_filt1 = data1.values.tolist()
    HeatMap(data=data_filt1, use_local_extrema=False, min_opacity=0.5,max_opacity=0.95,radius=15)\
        .add_to(folium.FeatureGroup(name=str(imei) + ' speed').add_to(map_speed))

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
