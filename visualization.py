from folium import folium
from collections import defaultdict
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd
import branca.colormap as cm
import os
from collections import defaultdict
import folium as folium


def visualize_drives(drives_by_imei_dictionary, cells_per_drive_per_modem_avg):
    for key in drives_by_imei_dictionary.keys():
        locations_in_drive_per_modem = cells_per_drive_per_modem_avg[key][['lat', 'lon']]
        locations_to_list = locations_in_drive_per_modem.values.tolist()
        modems_holder, stat_holder = {}, {}
        map_speed_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=10)  # Create a map object
        map_switchover_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073],
                                                        zoom_start=10)  # Create a map object
        map_loss_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=10)  # Create a map object
        map_rsrp_for_drive_for_modem = folium.Map(location=[32.17752, 34.93073], zoom_start=10)  # Create a map object

        # making the cells according to range
        all_cells_df = cells_per_drive_per_modem_avg[key].reset_index()
        for index, row in all_cells_df.iterrows():
            folium.Circle((row['lat'], row['lon']), radius=row['range'], fill=True, opacity=0.9, weight=2).add_to(
                map_switchover_for_drive_for_modem)

        drives_by_imei_dictionary[key]['latitude_perimeter_shift'] = drives_by_imei_dictionary[key][
            'latitude_perimeter'].shift()
        drives_by_imei_dictionary[key]['longitude_perimeter_shift'] = drives_by_imei_dictionary[key][
            'longitude_perimeter'].shift()
        # Calculate the Euclidean distance between the two points for each row
        drives_by_imei_dictionary[key]['distance'] = drives_by_imei_dictionary[key].apply(euclidean_distance, axis=1)
        drives_by_imei_dictionary[key]['speed'] = drives_by_imei_dictionary[key]['distance'] * 60 * 60 * 111 * -1
        drives_by_imei_dictionary[key].fillna(0, inplace=True)

        colormap = cm.linear.RdYlGn_11.scale(0, 35).to_step(20)
        colormap.caption = 'Green is greater velocity'
        map_speed_for_drive_for_modem.add_child(colormap)
        data1 = drives_by_imei_dictionary[key][['latitude_perimeter', 'longitude_perimeter', 'speed']]
        filtered_data_1 = data1.values.tolist()
        HeatMap(data=filtered_data_1, use_local_extrema=False, min_opacity=0.5, max_opacity=0.95, radius=15) \
            .add_to(folium.FeatureGroup(name=str(key) + ' speed').add_to(map_speed_for_drive_for_modem))

        data2 = drives_by_imei_dictionary[key][['latitude_perimeter', 'longitude_perimeter', 'switchover']]
        data2 = data2[data2['switchover'] == 1]
        marker_cluster = MarkerCluster(name=str(key) + ' cluster').add_to(map_switchover_for_drive_for_modem)
        for index, row in data2.iterrows():
            folium.Marker(location=[row['latitude_perimeter'], row['longitude_perimeter']],
                          popup=row['switchover']).add_to(marker_cluster)

        filtered_data_2 = data2.values.tolist()
        HeatMap(data=filtered_data_2, use_local_extrema=False, min_opacity=0.0, max_opacity=0.95, radius=10).add_to(
            folium.FeatureGroup(name=str(key) + ' switchover').add_to(map_switchover_for_drive_for_modem))

        data3 = drives_by_imei_dictionary[key][['latitude_perimeter', 'longitude_perimeter', 'loss_rate']]
        data3 = data3[data3['loss_rate'] > 0]
        marker_cluster_loss = MarkerCluster(name=str(key) + ' cluster').add_to(map_loss_for_drive_for_modem)
        for index, row in data3.iterrows():
            folium.Marker(location=[row['latitude_perimeter'], row['longitude_perimeter']],
                          popup=row['loss_rate']).add_to(marker_cluster_loss)
        filtered_data_3 = data3.values.tolist()
        HeatMap(data=filtered_data_3, use_local_extrema=False, min_opacity=0.5,
                max_opacity=0.9,
                radius=25).add_to(folium.FeatureGroup(name=str(key) + ' loss').add_to(map_loss_for_drive_for_modem))

        data4 = drives_by_imei_dictionary[key][['latitude_perimeter', 'longitude_perimeter', 'rsrp']]
        filtered_data_4 = data4.values.tolist()
        HeatMap(data=filtered_data_4, use_local_extrema=False, min_opacity=0.5,
                max_opacity=0.9,
                radius=25).add_to(folium.FeatureGroup(name=str(key) + ' rsrp').add_to(map_rsrp_for_drive_for_modem))

        # corr_mat = modems_holder[imei].corr()
        # corr_mat.to_csv(str(imei) + '_corr_mat.csv')
        # stat_holder[imei] = modems_holder[imei].describe()[['switchover', 'loss_rate', 'rsrp', 'latency_mean']].loc[
        # 'mean']
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
        # pd.DataFrame.from_dict(stat_holder).to_csv(key + '-stats_compare.csv')
        x = 5


def prepare_distance_to_cells(drives_dict, cells_location_dict):
    rsrp_dict = {}
    for key in drives_dict.keys():
        drives_dict[key].reset_index()
        drives_dict[key].insert(18, "celllat", 0, allow_duplicates=False)
        drives_dict[key].insert(18, "celllong", 0, allow_duplicates=False)
        count = 0
        for i, row in drives_dict[key].iterrows():
            cell_lat, cell_long = cells_location_dict.get(row['globalcellid'], (0, 0))
            if cell_long == 0 and cell_lat == 0:
                count = count + 1
            drives_dict[key].at[i, 'celllat'] = cell_lat
            drives_dict[key].at[i, 'celllong'] = cell_long
        drives_dict[key] = drives_dict[key].assign(
            disttocell=lambda x: euclidean_distance_points(x['latitude'], x['longitude'], x['celllat'],
                                                           x['celllong']))
        # drives_dict[key].at[i, 'dist2cell'] = haversine(row['longitude'], row['latitude'], row['celllong'],
        #                                                 row['celllat'])
        # drives_dict[key] = drives_dict[key].assign(
        #     dist2cell=lambda x: euclidean_distance_points(x['latitude'], x['longitude'], x['celllat'],
        #                                                        x['celllong']))
        rsrp_dict[key] = drives_dict[key][drives_dict[key]['celllong'] > 0][['rsrp', 'dist2cell']]
        rsrp_dict[key].plot(x="dist2cell", y=["rsrp"], kind="line", figsize=(10, 10))
        # cell_calculation.calculate_switchover(drives_dict[key])  # Calculate switch over per drive per imei
        mp.show()
        x = 5
    return drives_dict, rsrp_dict
