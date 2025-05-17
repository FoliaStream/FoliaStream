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

    # Generate map
    map_lat = statistics.mean([sink[sink_lat].mean(),source[source_lat].mean()])
    map_lon = statistics.mean([sink[sink_lon].mean(),source[source_lon].mean()])
    map_coords = (map_lat, map_lon)
    map = folium.Map(map_coords, zoom_start=4)

    # Generate markers
    for i, rsink in sink.iterrows():
        folium.Marker(
            location = [rsink[sink_lat], rsink[sink_lon]], 
            popup=str(rsink[sink_id]),
            icon = folium.Icon(color="blue", icon="")
        ).add_to(map)

    for j, rsource in source.iterrows():
        folium.Marker(
            location = [rsource[source_lat], rsource[source_lon]], 
            popup=str(rsource[source_id]),
            icon = folium.Icon(color="red", icon="")
        ).add_to(map)

    # Tiles
    folium.TileLayer('openstreetmap').add_to(map)

    # Legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="position: fixed; 
        top: 50px; left: 50px; width: 200px; height:95px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white; opacity: 0.6;">
        &nbsp; <b>Legend</b> <br>
        &nbsp; Sink node &nbsp; <i class="fa fa-circle" style="color:blue"></i><br>
        &nbsp; Source node &nbsp; <i class="fa fa-circle" style="color:red"></i><br>
    </div>
    {% endmacro %}
    '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)
    map.get_root().add_child(legend)

    return map



# STEP . Create matrix

def create_matrix(source, sink, source_id, source_lat, source_lon, sink_id, sink_lat, sink_lon, emission_cost, capture_cost):


    # Distance matrix

    distance_matrix = pd.DataFrame()

    for i, rsink in sink.iterrows():

        for j, rsource in source.iterrows():

            distance_matrix.at[i,j] = geodesic((float(rsink[sink_lat]), float(rsink[sink_lon])), (float(rsource[source_lat]), float(rsource[source_lon]))).km
    
    distance_matrix = distance_matrix.set_index(sink[sink_id])

    distance_matrix = distance_matrix.rename(columns = source[source_id])

    # Cost matrix
    cost_matrix = pd.DataFrame(distance_matrix)

    for col in cost_matrix.columns:

        for id in cost_matrix[col].index:

            if cost_matrix.at[id, col] < 180:
                cost_matrix.at[id, col] = cost_matrix.at[id, col]*0.01417 + capture_cost

            elif cost_matrix.at[id, col] >= 180 and cost_matrix.at[id, col] < 500:
                cost_matrix.at[id, col] = cost_matrix.at[id, col]*0.01196 + capture_cost

            elif cost_matrix.at[id, col] >= 500 and cost_matrix.at[id, col] < 750:
                cost_matrix.at[id, col] = cost_matrix.at[id, col]*0.01147 + capture_cost

            elif cost_matrix.at[id, col] >= 750 and cost_matrix.at[id, col] < 1500:
                cost_matrix.at[id, col] = cost_matrix.at[id, col]*0.01139 + capture_cost

            else:
                cost_matrix.at[id, col] = cost_matrix.at[id, col]*0.01132 + capture_cost
    
    new_row = pd.DataFrame({col:emission_cost for col in cost_matrix.columns}, index=["Atmosphere"])

    cost_matrix = pd.concat([cost_matrix,new_row], axis=0)

    return cost_matrix

