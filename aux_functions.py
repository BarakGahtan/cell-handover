import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from numpy import sin, cos, arccos, pi, round


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
    meters = 6371 * c * 1000
    return meters


def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees


def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2):
    if latitude1 == 0 or latitude2 == 0 or longitude1 == 0 or longitude2 == 0:
        return 0
    theta = longitude1 - longitude2
    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) +
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        )
    )
    return round(distance * 1.609344, 2) * 1000
