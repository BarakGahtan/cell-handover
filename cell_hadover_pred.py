import os
from collections import defaultdict
import folium as folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd
import branca.colormap as cm
import cell_calculation
from aux_functions import euclidean_distance
from cell_calculation import get_unique_cells_in_drive

NUM_DRIVES = 2
picke_name = 'pickle_rick.pkl'
big_df = pd.read_pickle(picke_name)
drives = [v for k, v in big_df.groupby(['date', 'time'])]  # filter according to start of the drive. unique.

# Now filter according to modem in each drive
drives_by_modem = []
for i in range(len(drives)):
    drives_by_modem.append([v for k, v in drives[i].groupby('modem_id')])
# drive_by_modems is list of lists. each drive is a list of lists. the list inside is for diff modems.

# each drive is broken into 5-6 modems. There are 600 drives and at most 6 modems.
drives_by_modem_dict = {}
for i in range(NUM_DRIVES):  # len(drives_by_modem)
    for j in range(len(drives_by_modem[i])):  # len(drives_by_modem[i]) - number os modems in drive.
        key_for_dict = str(drives_by_modem[i][j]['date'].iloc[0] + '_' + drives_by_modem[i][j]['time'].iloc[0] + '_' +
                           drives_by_modem[i][j]['modem_id'].iloc[0])
        drives_by_modem_dict[key_for_dict] = drives_by_modem[i][j]

# identifying the cells of each drive per modem. the key of the dict is when the drive started following by the
# modem_id. key=date_time_modemID
cells_per_drive_per_modem_avg = {}
for key in drives_by_modem_dict.keys():  # len(drives_by_modem)
    cells_per_drive_per_modem_avg[key] = pd.concat(get_unique_cells_in_drive(drives_by_modem_dict[key]), axis=0,
                                                   join='outer').dropna(axis=0)

# Create a map object
map_object = folium.Map(location=[32.17752, 34.93073], zoom_start=15)

for key in drives_by_modem_dict.keys():
    locations_in_drive_per_modem = cells_per_drive_per_modem_avg[key][['lat', 'lon']]
    locations_to_list = locations_in_drive_per_modem.values.tolist()
    unique_IMEIs_per_drive_per_modem = drives_by_modem_dict[key]['imei'].unique()
    modems_holder, stat_holder = {}, {}
    map_speed_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=25)
    map_switchover_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=25)
    map_loss_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=25)
    map_rsrp_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=25)

    # making the cells according to range
    all_cells_df = cells_per_drive_per_modem_avg[key].reset_index()
    for index, row in all_cells_df.iterrows():
        folium.Circle((row['lat'], row['lon']), radius=row['range'], fill=True, opacity=0.9, weight=2).add_to(map_switchover_for_drive_for_modem)
    for imei in unique_IMEIs_per_drive_per_modem:
        modems_holder[imei] = drives_by_modem_dict[key][drives_by_modem_dict[key]['imei'] == imei]
        # modems_holder[imei].insert(18, "switchover", 0, allow_duplicates=False)
        # modems_holder[imei].replace(999999, np.nan, inplace=True)
        # modems_holder[imei].ffill(inplace=True)
        # modems_holder[imei].bfill(inplace=True)
        # modems_holder[imei]['globacellid_shift'] = modems_holder[imei]['globalcellid'].shift()
        # modems_holder[imei]['switchover'] = modems_holder[imei].apply(
        #     lambda x: 0 if x['globalcellid'] == x['globacellid_shift'] else 1, axis=1)
        modems_holder[imei]['latitude_perimeter_shift'] = modems_holder[imei]['latitude_perimeter'].shift()
        modems_holder[imei]['longitude_perimeter_shift'] = modems_holder[imei]['longitude_perimeter'].shift()
        # Calculate the Euclidean distance between the two points for each row
        modems_holder[imei]['distance'] = modems_holder[imei].apply(euclidean_distance, axis=1)
        modems_holder[imei]['speed'] = modems_holder[imei]['distance'] * 60 * 60 * 111 * -1
        modems_holder[imei].fillna(0, inplace=True)
        cell_calculation.calculate_switchover(modems_holder[imei])

        colormap = cm.linear.RdYlGn_11.scale(0, 35).to_step(20)
        colormap.caption = 'Green is greater velocity'
        map_speed_for_drive_for_modem.add_child(colormap)
        data1 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'speed']]
        data_filt1 = data1.values.tolist()
        HeatMap(data=data_filt1, use_local_extrema=False, min_opacity=0.5, max_opacity=0.95, radius=15) \
            .add_to(folium.FeatureGroup(name=str(imei) + ' speed').add_to(map_speed_for_drive_for_modem))

        # data2 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'switchover']].groupby([
        # 'latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
        data2 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'switchover']]
        data2 = data2[data2['switchover'] == 1]
        marker_cluster = MarkerCluster(name=str(imei) + ' cluster').add_to(map_switchover_for_drive_for_modem)
        for index, row in data2.iterrows():
            folium.Marker(location=[row['latitude_perimeter'], row['longitude_perimeter']],
                          popup=row['switchover']).add_to(marker_cluster)

        data_filt2 = data2.values.tolist()
        HeatMap(data=data_filt2, use_local_extrema=False, min_opacity=0.0,max_opacity=0.95,radius=10).add_to(
            folium.FeatureGroup(name=str(imei) + ' switchover').add_to(map_switchover_for_drive_for_modem))

        # data3 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'loss_rate']].groupby([
        # 'latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
        data3 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'loss_rate']]
        data3 = data3[data3['loss_rate'] > 0]

        marker_cluster_loss = MarkerCluster(name=str(imei) + ' cluster').add_to(map_loss_for_drive_for_modem)
        for index, row in data3.iterrows():
            folium.Marker(location=[row['latitude_perimeter'], row['longitude_perimeter']],
                          popup=row['loss_rate']).add_to(marker_cluster_loss)
        data_filt3 = data3.values.tolist()
        HeatMap(data=data_filt3, use_local_extrema=False, min_opacity=0.5,
                max_opacity=0.9,
                radius=25).add_to(folium.FeatureGroup(name=str(imei) + ' loss').add_to(map_loss_for_drive_for_modem))

        # data4 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'rsrp']].groupby([
        # 'latitude_perimeter', 'longitude_perimeter']).mean().reset_index()
        data4 = modems_holder[imei][['latitude_perimeter', 'longitude_perimeter', 'rsrp']]
        data_filt4 = data4.values.tolist()
        HeatMap(data=data_filt4, use_local_extrema=False, min_opacity=0.5,
                max_opacity=0.9,
                radius=25).add_to(folium.FeatureGroup(name=str(imei) + ' rsrp').add_to(map_rsrp_for_drive_for_modem))

        corr_mat = modems_holder[imei].corr()
        corr_mat.to_csv(str(imei) + '_corr_mat.csv')
        stat_holder[imei] = modems_holder[imei].describe()[['switchover', 'loss_rate', 'rsrp', 'latency_mean']].loc[
            'mean']
        # corr_mat = modems_holder[imei].describe()
        # corr_mat.to_csv(str(imei) + '_stats.csv')
        folium.LayerControl().add_to(map_speed_for_drive_for_modem)
        map_speed_for_drive_for_modem.save(key + '-speed.html')
        folium.LayerControl().add_to(map_switchover_for_drive_for_modem)
        map_switchover_for_drive_for_modem.save(key + '-switchover.html')
        folium.LayerControl().add_to(map_loss_for_drive_for_modem)
        map_loss_for_drive_for_modem.save(key + '-loss.html')
        folium.LayerControl().add_to(map_rsrp_for_drive_for_modem)
        map_rsrp_for_drive_for_modem.save(key + '-rsrp.html')
        pd.DataFrame.from_dict(stat_holder).to_csv(key + '-stats_compare.csv')
        x = 5
# print(drive_df)
