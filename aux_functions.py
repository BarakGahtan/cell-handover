import pandas as pd
import numpy as np


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
