import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


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


def euclidean_distance_points(x1, y1, x2, y2):
    x =5
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km
