import os
import errno
import requests
import statistics
import folium
import branca


import pandas as pd
import numpy as np

from loguru import logger
from geopy.distance import geodesic
from pipe.functions.functions_II import request_url

# //////////////////////////////////////////////////////
#                  FUNCTIONS I LEVEL
# //////////////////////////////////////////////////////

#----------------------
# STEP . Create folder
#----------------------

def create_folder(path: str):

    # Check exist
    if not os.path.exists(path):
        # Create folder
        try:
            logger.info(f"create folder: {path}")
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    
    return path


# STEP . Source load 

# Import source data from API
def source_import_api(url, params):

    # Merge url and params in url for query
    url_query = request_url(url, params)

    # Perform request
    response = requests.get(url_query)

    # if response == 200:
        # Success
    data = response.json()['assets']

    return data
    

# Convert api response to dataframe
def source_edit(source, id_col, emit_col, lat_col, lon_col):

    df_source = pd.DataFrame()

    for i in range(len(source)):
        if source[i]['Id'] is not None:
            df_source.at[i,id_col] = source[i]["Id"]
            df_source.at[i,emit_col] = float(source[i]['EmissionsSummary'][0]['EmissionsQuantity'])
            df_source.at[i,lat_col] = float(source[i]['Centroid']['Geometry'][1])
            df_source.at[i,lon_col] = float(source[i]['Centroid']['Geometry'][0])

    return df_source


# STEP . Sink load

# Import csv
def csv_import(csv_path):

    df = pd.read_csv(csv_path)

    return df

# Edit data
def sink_edit(sink, id_col, country_col, capacity_col, lat_col, lon_col, country):

    # Filter country
    sink_out = sink[sink[country_col] == country] 

    # Filter necessary columns
    sink_out = sink_out[[id_col, capacity_col, lat_col, lon_col]]
    sink_out[id_col] = sink_out[id_col].astype(float)
    
    return sink_out


# STEP . Nodes map

def nodes_map(source, sink, source_id, source_lat, source_lon, sink_id, sink_lat, sink_lon):

    map_lat = statistics.mean([sink[sink_lat].mean(),source[source_lat].mean()])
    map_lon = statistics.mean([sink[sink_lon].mean(),source[source_lon].mean()])

    map_coords = (map_lat, map_lon)

    map = folium.Map(map_coords, zoom_start=4)
    
    return map

