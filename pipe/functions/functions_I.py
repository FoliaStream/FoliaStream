import os
import errno
import requests
import statistics
import folium
import branca
import pulp

import cvxpy as cp
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



# STEP . Network optimization

def network_optimization(df_source, df_sink, df_cost_matrix, source_id, sink_id, source_capacity, sink_capacity):

    df_cost_matrix = df_cost_matrix.rename(columns={"Unnamed: 0":sink_id})

    # Network initialization
    network = pulp.LpProblem("Network_problem", pulp.LpMinimize)

    # Generate source and sink lists 
    source_list = df_source[source_id].astype(str)
    sink_list = list(df_sink[sink_id].astype(str))
    sink_list.append("Atmosphere")  


    # Set sink_id as index of transport cost matrix
    transport_cost = df_cost_matrix.set_index(sink_id)

    # Create transport cost dictionary
    transport_dict = {(source_id, sink_id): transport_cost.loc[sink_id,source_id] 
                  for sink_id in transport_cost.index.astype(str)
                  for source_id in transport_cost.columns}

    # Demand and Supply
    demand = dict(zip(df_sink[sink_id].astype(str), df_sink[sink_capacity]))
    atmo_demand = {"Atmosphere":100000000000000000000000000000000000000000000000000000}
    demand.update(atmo_demand)
    supply = dict(zip(df_source[source_id].astype(str), df_source[source_capacity]))

    # Create decision variables for co2 transportation manually
    co2 = {}
    for i in source_list:
        for j in sink_list:
            co2[i, j] = pulp.LpVariable(f"route_{i}_{j}", lowBound=0, cat="Continuous")

    # Objective function (minimizing transport cost)
    network += pulp.lpSum(transport_dict[(i, j)] * co2[i, j] for i in source_list for j in sink_list), "Total_Transportation_Cost"


    # Demand constraints (one per sink)
    for j in sink_list:
        network += pulp.lpSum(co2[i, j] for i in source_list) <= demand[j], f"Demand_Constraint_{j}"


    # Supply constraints (one per source)
    for i in source_list: 
        network += pulp.lpSum(co2[i, j] for j in sink_list) >= supply[i], f"Supply_Constraint_{i}"

    network.solve()
    results = []

    # Extract the decision variable values
    for i in source_list:
        for j in sink_list:
            # Get the value of the decision variable (amount of CO2 transported from source i to sink j)
            amount = co2[i, j].varValue

            if amount > 0:  # You can choose to only include routes with non-zero flow
                results.append({
                    'source_id' : i,
                    'sink_id': j,
                    'co2_transported': amount
                })

    # Convert the results list to a Pandas DataFrame
    df_results = pd.DataFrame(results)

    return df_results




# STEP . Network map 
def network_map(network, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon):

    sink = sink.rename(columns={sink_id:f'sink_{sink_id}'})
    source = source.rename(columns={source_id:f'source_{source_id}'})

    source_id = f'source_{source_id}'
    sink_id = f'sink_{sink_id}'



    network[source_lat] = pd.merge(network, source, on=source_id, how="inner")[source_lat]
    network[source_lon] = pd.merge(network, source, on=source_id, how="inner")[source_lon]

    network[sink_lat] = pd.Series()
    network[sink_lon] = pd.Series()

    for i, row in network.iterrows():
        if row[sink_id] == "Atmosphere":
            network.at[i,sink_lat] = row[source_lat]
            network.at[i,sink_lon] = row[source_lon]
        else:
            network.at[i,sink_lat] = sink[sink[sink_id] == int(float(row[sink_id]))][sink_lat]
            network.at[i,sink_lon] = sink[sink[sink_id] == int(float(row[sink_id]))][sink_lon]

    links = []

    map_lat = statistics.mean([sink[sink_lat].mean(),source[source_lat].mean()])
    map_lon = statistics.mean([sink[sink_lon].mean(),source[source_lon].mean()])

    map_coords = (map_lat, map_lon)

    map = folium.Map(map_coords, zoom_start=4)

    for i,row in network.iterrows():
        if row[sink_id] == "Atmosphere":
            links.append([[row[source_lat], row[source_lon]], [row[source_lat], row[source_lon]], str(f"{row[source_id]}__{row[sink_id]}")])
        else:
            links.append([[row[source_lat], row[source_lon]], [row[sink_lat], row[sink_lon]], str(f"{row[source_id]}__{row[sink_id]}")])

    for i in range(len(links)):
        line = folium.PolyLine(locations=[links[i][0], links[i][1]], popup=links[i][2])
        line.add_to(map)

    for i,row in network.iterrows():
        folium.Marker(
            location = [row[sink_lat], row[sink_lon]],
            popup=str(row[sink_id]),
            icon = folium.Icon(color="blue", icon="")
        ).add_to(map)

    for i,row in network.iterrows():
        folium.Marker(
            location = [row[source_lat], row[source_lon]],
            popup=str(row[source_id]),
            icon = folium.Icon(color="red", icon="")
        ).add_to(map)

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