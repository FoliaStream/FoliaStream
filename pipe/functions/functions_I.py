import os
import errno
import requests
import statistics
import folium
import branca


import pandas as pd
import numpy as np
import streamlit as st

from loguru import logger
from geopy.distance import geodesic
from elasticsearch import Elasticsearch, helpers
from pipe.functions.functions_II import export_data_structure, request_url

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



#------------------
# STEP . Sink load
#------------------

# Sink import
def sink_import(path):

    # Read from csv
    sink_in = pd.read_csv(path)

    return sink_in

# Sink edit
def sink_edit(sink_in, id_col, country_col, capacity_col, lat_col, lon_col, country):

    # Filter country
    sink_data = sink_in[sink_in[country_col] == country]

    # Filter necessary columns
    sink_data = sink_data[[id_col, capacity_col, lat_col, lon_col]]
    sink_data[id_col] = sink_data[id_col].astype(float)

    return sink_data


# Sink export
def sink_export(sink, host, auth, index, mappings, id_col):

    # Host setup
    # es = Elasticsearch(hosts=host) #,basic_auth=auth)
    es = Elasticsearch(
        [st.secrets['ES_HOST']],
        http_auth=(st.secrets['ES_USER'], st.secrets['ES_PASSWORD'])
    )

    try:
        es.indices.delete(index=index, ignore=[400,404])
    except:
        pass

    # Index creation
    es.indices.create(index=index, body=mappings)

    # Data setup
    actions_in = export_data_structure(sink, index, id_col)

    # Export
    success, fail = helpers.bulk(es, actions_in)

    return success, fail



#--------------------
# STEP . Source load
#--------------------

# Source import
def source_import(url, params):

    # Merge url and params in url for query
    url_query = request_url(url, params)

    # Perform request
    response = requests.get(url_query)

    # if response == 200:
        # Success
    data = response.json()['assets']

    return data

# Source edit
def source_edit(source, id_col, emit_col, lat_col, lon_col):

    # From json to df

    # df_source = pd.json_normalize(source)
    df_source = pd.DataFrame()

    for i in range(len(source)):
        if source[i]['Id'] is not None:
            df_source.at[i,id_col] = source[i]["Id"]
            df_source.at[i,emit_col] = float(source[i]['EmissionsSummary'][0]['EmissionsQuantity'])
            df_source.at[i,lat_col] = float(source[i]['Centroid']['Geometry'][1])
            df_source.at[i,lon_col] = float(source[i]['Centroid']['Geometry'][0])

    return df_source


# Source export
def source_export(source, host, auth, index, mappings, id_col):

    # Host index setup 
    es = Elasticsearch(hosts=host)#, basic_auth=auth)
    try:
        es.indices.delete(index=index, ignore=[400,404])
    except:
        pass


    # Index creation
    es.indices.create(index=index, body=mappings)

    # Data setup
    actions_in = export_data_structure(source, index, id_col)

    # Export
    success, fail = helpers.bulk(es, actions_in)

    return success, fail


#------------------
# STEP . Nodes map
#------------------

# Import source and sink from localhost
def import_data(host, auth, index, query, columns):

    # Host setup
    es = Elasticsearch(hosts=host)#, basic_auth=auth)
    es.indices.refresh(index=index)

    # Search
    response = es.search(index=index, body=query)
    documents = response['hits']['hits']

    # Create df results
    columns_dict_list = {}

    for col in columns:
        columns_dict_list[str(col)] = []

    for doc in documents:
        if index == 'source':
            for col in columns_dict_list.keys():
                columns_dict_list[col].append(doc[f'_{index}'][col])
        
        else:
            for col in columns_dict_list.keys():
                columns_dict_list[col].append(doc['_source'][f'_{index}'][col])
    
    results = pd.DataFrame(columns_dict_list)

    return results

# Create nodes map
def nodes_map(source, sink, sink_lat, sink_lon, sink_id, source_lat, source_lon, source_id):

    # Identify center coordinates
    center_lat = statistics.mean([sink[sink_lat].mean(), source[source_lat].mean()])
    center_lon = statistics.mean([sink[sink_lon].mean(), source[source_lon].mean()])
    center_coords = (center_lat, center_lon)

    # Initialize map
    map = folium.Map(center_coords, zoom_start=4)

    # Add sinks
    for _, rsink in sink.iterrows():
        folium.Marker(
            location = [rsink[sink_lat], rsink[sink_lon]],
            popup = str(rsink[sink_id]),
            icon = folium.Icon(color='blue', icon='')
        ).add_to(map)

    # Add sources
    for _, rsource in source.iterrows():
        folium.Marker(
            location = [rsource[source_lat], rsource[source_lon]],
            popup = str(rsource[source_id]),
            icon = folium.Icon(color='red', icon='')
        ).add_to(map)

    # Tiles
    folium.TileLayer('openstreetmap').add_to(map)

    # Legend
    legend_html = '''
    {% macro html(this, kwargs) %}
    <div style="position: fixed; 
        top: 50px; left: 50px; width: 200px; height:75px; 
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