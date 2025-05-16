
import os 
import errno
import requests
import statistics
import folium
import branca
import time
import pulp

import pandas as pd
import numpy as np

from loguru import logger
from pipe.functions.functions_II import request_url, export_data_structure, elbow_method, create_clusters, export_data_structure_cluster, export_data_structure_centroids
from elasticsearch import Elasticsearch, helpers
from geopy.distance import geodesic


# ////////////////////////////////////////////////
#                   FUNCTIONS I
# ////////////////////////////////////////////////



# STEP . Create folder 

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


# Export source dataframe to elastichost
def source_export(source, host, auth, index, mappings, id_col, emit_col, lat_col, lon_col, cluster_col = None):

    # Host index setup
    es = Elasticsearch(hosts=host, basic_auth=auth)
    es.indices.delete(index=index, ignore=[400,404])

    # Index creation
    es.indices.create(index=index, body=mappings)
    
    # Data setup
    if cluster_col is None:
        actions_in = export_data_structure(source, index, id_col, emit_col, lat_col, lon_col)
    else: 
        actions_in = export_data_structure_cluster(source, index, id_col, emit_col, lat_col, lon_col, cluster_col)

    # Export
    success, fail = helpers.bulk(es, actions_in)

    return  success, fail


# STEP . Create cluster
def create_cluster(source, lat_col, lon_col, capacity_col, cluster_col):

    # Calculate number of clusters
    elbow_point = elbow_method(source, lat_col, lon_col)

    # Calculate clusters
    centroids, labels = create_clusters(source, lat_col, lon_col, elbow_point, capacity_col)

    # Indicate cluster on source
    source[cluster_col] = pd.Series(labels)

    return centroids, source



def centroids_export(centroids, host, auth, index, mappings, id_col, lat_col, lon_col):

    # Host index setup
    es = Elasticsearch(hosts=host) #, basic_auth=auth)
    es.indices.delete(index=index, ignore=[400,404])

    # Index creation
    es.indices.create(index=index, body=mappings)

    # Data setup
    actions_in = export_data_structure_centroids(centroids, index, id_col, lat_col, lon_col)

    # Export
    success, fail = helpers.bulk(es, actions_in)

    return success, fail





# STEP . Sink load

def sink_edit(sink, id_col, country_col, capacity_col, lat_col, lon_col, country):

    # Filter country
    sink_out = sink[sink[country_col] == country] 

    # Filter necessary columns
    sink_out = sink_out[[id_col, capacity_col, lat_col, lon_col]]
    sink_out[id_col] = sink_out[id_col].astype(float)
    
    return sink_out





# Export sink dataframe to elastichost
def sink_export(sink, host, auth, index, mappings, id_col, capacity_col, lat_col, lon_col):

    # Host index setup 
    es = Elasticsearch(hosts=host)#, basic_auth=auth)
    es.indices.delete(index=index, ignore=[400,404])

    # Index creation
    es.indices.create(index=index, body=mappings)

    # Data setup
    actions_in = export_data_structure(sink, index, id_col, capacity_col, lat_col, lon_col)

    # Export
    success, fail = helpers.bulk(es, actions_in)

    return success, fail


# STEP . Nodes map

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




# Nodes map

def nodes_map(df_source, df_sink, sink_lat, sink_lon, sink_id, source_lat, source_lon, source_id, df_centroids = None, centroids_id = None, centroids_lat = None, centroids_lon = None):
        
    map_lat = statistics.mean([df_sink[sink_lat].mean(),df_source[source_lat].mean()])
    map_lon = statistics.mean([df_sink[sink_lon].mean(),df_source[source_lon].mean()])

    map_coords = (map_lat, map_lon)

    map = folium.Map(map_coords, zoom_start=4)

    if df_centroids is None:
        
        for i, rsink in df_sink.iterrows():
            folium.Marker(
                location = [rsink[sink_lat], rsink[sink_lon]], 
                popup=str(rsink[sink_id]),
                icon = folium.Icon(color="blue", icon="")
            ).add_to(map)

        for j, rsource in df_source.iterrows():
            folium.Marker(
                location = [rsource[source_lat], rsource[source_lon]], 
                popup=str(rsource[source_id]),
                icon = folium.Icon(color="red", icon="")
            ).add_to(map)

        folium.TileLayer('openstreetmap').add_to(map)

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

    else:

        for i, rsink in df_sink.iterrows():
            folium.Marker(
                location = [rsink[sink_lat], rsink[sink_lon]], 
                popup=str(rsink[sink_id]),
                icon = folium.Icon(color="blue", icon="")
            ).add_to(map)

        for j, rsource in df_source.iterrows():
            folium.Marker(
                location = [rsource[source_lat], rsource[source_lon]], 
                popup=str(rsource[source_id]),
                icon = folium.Icon(color="red", icon="")
            ).add_to(map)
        
        for k, rcentroid in df_centroids.iterrows():
            folium.Marker(
                location = [rcentroid[centroids_lat], rcentroid[centroids_lon]],
                popup=str(rcentroid[centroids_id]),
                icon = folium.Icon(color='green', icon="")
            ).add_to(map)

        folium.TileLayer('openstreetmap').add_to(map)

        legend_html = '''
        {% macro html(this, kwargs) %}
        <div style="position: fixed; 
            top: 50px; left: 50px; width: 200px; height:95px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; opacity: 0.6;">
            &nbsp; <b>Legend</b> <br>
            &nbsp; Sink node &nbsp; <i class="fa fa-circle" style="color:blue"></i><br>
            &nbsp; Source node &nbsp; <i class="fa fa-circle" style="color:red"></i><br>
            &nbsp; Centroid &nbsp; <i class="fa fa-circle" style="color:green"></i><br>
        </div>
        {% endmacro %}
        '''
        legend = branca.element.MacroElement()
        legend._template = branca.element.Template(legend_html)
        map.get_root().add_child(legend)

        return map



# STEP . Create Matrix

def create_matrix(df_source, df_sink, source_lat, source_lon, source_id, sink_lat, sink_lon, sink_id, capture_cost, emission_cost, df_centroids = None, centroid_id = None, centroid_lat = None, centroid_lon = None, source_cluster = None):


    if df_centroids is None:

        # Distance matrix

        source_sink_distance = pd.DataFrame()

        for i, rsink in df_sink.iterrows():

            for j, rsource in df_source.iterrows():

                source_sink_distance.at[i,j] = geodesic((float(rsink[sink_lat]), float(rsink[sink_lon])), (float(rsource[source_lat]), float(rsource[source_lon]))).km
        
        source_sink_distance = source_sink_distance.set_index(df_sink[sink_id])

        source_sink_distance = source_sink_distance.rename(columns = df_source[source_id])

        # Cost matrix

        source_sink_pipeline_capex_opex_cost = pd.DataFrame(source_sink_distance)

        for col in source_sink_pipeline_capex_opex_cost.columns:

            for id in source_sink_pipeline_capex_opex_cost[col].index:

                if source_sink_pipeline_capex_opex_cost.at[id, col] < 180:
                    source_sink_pipeline_capex_opex_cost.at[id, col] = source_sink_pipeline_capex_opex_cost.at[id, col]*0.01417 + capture_cost

                elif source_sink_pipeline_capex_opex_cost.at[id, col] >= 180 and source_sink_pipeline_capex_opex_cost.at[id, col] < 500:
                    source_sink_pipeline_capex_opex_cost.at[id, col] = source_sink_pipeline_capex_opex_cost.at[id, col]*0.01196 + capture_cost

                elif source_sink_pipeline_capex_opex_cost.at[id, col] >= 500 and source_sink_pipeline_capex_opex_cost.at[id, col] < 750:
                    source_sink_pipeline_capex_opex_cost.at[id, col] = source_sink_pipeline_capex_opex_cost.at[id, col]*0.01147 + capture_cost

                elif source_sink_pipeline_capex_opex_cost.at[id, col] >= 750 and source_sink_pipeline_capex_opex_cost.at[id, col] < 1500:
                    source_sink_pipeline_capex_opex_cost.at[id, col] = source_sink_pipeline_capex_opex_cost.at[id, col]*0.01139 + capture_cost

                else:
                    source_sink_pipeline_capex_opex_cost.at[id, col] = source_sink_pipeline_capex_opex_cost.at[id, col]*0.01132 + capture_cost
        
        new_row = pd.DataFrame({col:emission_cost for col in source_sink_pipeline_capex_opex_cost.columns}, index=["Atmosphere"])

        source_sink_pipeline_capex_opex_cost = pd.concat([source_sink_pipeline_capex_opex_cost,new_row], axis=0)

        return source_sink_pipeline_capex_opex_cost

    else:

        # Distance matrix

        source_centroid_distance_matrix = pd.DataFrame() 
        
        for j, rsource in df_source.iterrows():

            for k, rcenter in df_centroids.iterrows():

                source_centroid_distance_matrix.at[j,k] = geodesic((float(rsource[source_lat]), float(rsource[source_lon])), (float(rcenter[centroid_lat]), float(rcenter[centroid_lon]))).km

        source_centroid_distance_matrix = source_centroid_distance_matrix.set_index(df_source[source_id])

        centroid_sink_distance_matrix = pd.DataFrame()
        
        for i, rsink in df_sink.iterrows():
            
            for k, rcenter in df_centroids.iterrows():

                centroid_sink_distance_matrix.at[i,k] = geodesic((float(rsink[sink_lat]), float(rsink[sink_lon])), (float(rcenter[centroid_lat]), float(rcenter[centroid_lon]))).km

        centroid_sink_distance_matrix = centroid_sink_distance_matrix.set_index(df_sink[sink_id])

        source_centroid_sink_distance_matrix = pd.concat([source_centroid_distance_matrix, centroid_sink_distance_matrix], axis=0)
        
        # Cost matrix

        for col in source_centroid_sink_distance_matrix.columns:

            for id in source_centroid_sink_distance_matrix[col].index:

                if source_centroid_sink_distance_matrix.at[id, col] < 180:
                    source_centroid_sink_distance_matrix.at[id, col] = source_centroid_sink_distance_matrix.at[id,col]*0.01417 + capture_cost

                elif source_centroid_sink_distance_matrix.at[id, col] >= 180 and source_centroid_sink_distance_matrix.at[id, col] < 500:
                    source_centroid_sink_distance_matrix.at[id, col] = source_centroid_sink_distance_matrix.at[id,col]*0.01196 + capture_cost

                elif source_centroid_sink_distance_matrix.at[id, col] >= 500 and source_centroid_sink_distance_matrix.at[id, col] < 750:
                    source_centroid_sink_distance_matrix.at[id, col] = source_centroid_sink_distance_matrix.at[id,col]*0.01147 + capture_cost

                elif source_centroid_sink_distance_matrix.at[id, col] >= 750 and source_centroid_sink_distance_matrix.at[id, col] < 1500:
                    source_centroid_sink_distance_matrix.at[id, col] = source_centroid_sink_distance_matrix.at[id,col]*0.01139 + capture_cost
                       
                else:
                    source_centroid_sink_distance_matrix.at[id, col] = source_centroid_sink_distance_matrix.at[id,col]*0.01132 + capture_cost

        source_centroid_sink_distance_matrix['Atmosphere'] = emission_cost
        atmosphere_row = pd.DataFrame({col:emission_cost for col in source_centroid_sink_distance_matrix.columns}, index=['Atmosphere'])
        source_centroid_sink_distance_matrix = pd.concat([source_centroid_sink_distance_matrix, atmosphere_row], axis=0)

        return source_centroid_sink_distance_matrix
        


# STEP . Matrix export

def matrix_export(matrix_data, host, auth, matrix_index):

    # Host setup
    es = Elasticsearch(hosts=host, request_timeout=10000000) #basic_auth=auth,
    es.indices.delete(index=matrix_index, ignore=[400,404])

    # Data setup
    matrix_data = pd.DataFrame(matrix_data.astype(np.float64))
    matrix_data.index = matrix_data.index.astype(str)

    index_body = {
        "mappings":{
            "properties":{
                "node":{"type":"keyword"}
            }
        }
    }

    for col in matrix_data.columns.to_list():
        index_body['mappings']['properties'][col] = {"type":"float"}
    
    es.indices.create(
        index=matrix_index,
        body=index_body
    )

    actions_matrix = []
    
    for id in matrix_data.index:
        if id is not None:
            actions_matrix.append(
                {
                    '_op_type':'index',
                    '_index':matrix_index,
                    '_id':id,
                    str(f'_{matrix_index}'):{
                        'node':id
                    }
                }    
            )

    for i in range(len(actions_matrix)):
        for col in matrix_data.columns.to_list():
            actions_matrix[i][str(f'_{matrix_index}')][col] = matrix_data[col].loc[actions_matrix[i]['_id']]

    success, fail = helpers.bulk(es, actions_matrix)


    return success, fail






# Matrix import
def matrix_import(host, auth, index, query):

    # Host setup
    es = Elasticsearch(hosts=host)
    es.indices.refresh(index=index)

    # Search
    response = es.search(index=index, body=query)
    documents = response['hits']['hits']

    # Create df results
    
    list_docs = []

    for doc in documents:
        list_docs.append(doc['_source'][f'_{index}'])
    df_matrix = pd.DataFrame(list_docs)
    df_matrix = df_matrix.set_index(df_matrix['node'])
    df_matrix = df_matrix.drop(columns=['node'])


    return df_matrix



# Network optimization cluster

def network_opt_min_inter(source, sink, matrix, source_id, source_capacity, sink_id, sink_capacity, cluster_col):

    # breakpoint()


    # Network initialization
    network = pulp.LpProblem("Network_problem", pulp.LpMinimize)

    # Generate source and sink lists 
    source_list = source[source_id].astype(str)
    sink_list = list(sink[sink_id].astype(str))
    sink_list.append("Atmosphere")  
    clusters_list = matrix.columns.tolist()[:-1]



    transport_dict = {}
    for i,row in source.iterrows():
        transport_dict.update({(row[source_id],row[cluster_col]) : matrix.loc[str(row[source_id]), str(int(row[cluster_col]))]})

    transport_dict.update({(source_id,"Atmosphere"): matrix.loc[source_id,"Atmosphere"]
                           for source_id in source_list})
    
    transport_dict.update({(cluster_id, sink_id): matrix.loc[sink_id,cluster_id]
                        for cluster_id in clusters_list for sink_id in sink_list[:-1]})



    # breakpoint()

    return 





# Network optimization
def network_opt_min(source, sink, matrix, source_id, source_capacity, sink_id, sink_capacity):

    # Network initialization
    network = pulp.LpProblem("Network_problem", pulp.LpMinimize)

    # Generate source and sink lists 
    source_list = source[source_id].astype(str)
    sink_list = list(sink[sink_id].astype(str))
    sink_list.append("Atmosphere")  
    
    
    # Create transport cost dictionary
    transport_dict = {(source_id, sink_id): matrix.loc[sink_id,source_id] 
                  for sink_id in matrix.index.astype(str)
                  for source_id in matrix.columns}

    
    # Demand and Supply
    demand = dict(zip(sink[sink_id].astype(str), sink[sink_capacity]))
    atmo_demand = {"Atmosphere":100000000000000000000000000000000000000000000000000000}
    demand.update(atmo_demand)
    supply = dict(zip(source[source_id].astype(str), source[source_capacity]))

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
                    f'source_{source_id}' : i,
                    f'sink_{sink_id}': j,
                    'co2_transported': amount
                })

    # Convert the results list to a Pandas DataFrame
    df_results = pd.DataFrame(results)

    return df_results


# Network results export
def network_export(results, host, auth, index, path, source_id, sink_id):

    # Output as deliverable
    results.to_csv(path)

    # Export to host
    es = Elasticsearch(hosts=host) #, basic_auth=auth)
    es.indices.delete(index=index, ignore=[400,404])

    mappings_net = {
        'mappings':{
            'properties':{
                f'source_{source_id}':{'type':'float'},
                f'sink_{sink_id}':{'type':'float'},
                'co2_transported':{'type':'float'}
            }
        }
    }

    # Index creation
    es.indices.create(index=index, body=mappings_net)

    # Upload data
    actions_network = []
    for i in range(len(results)):
        if results.iloc[i]['co2_transported'] is not None:
            actions_network.append(
                {
                    '_op_type':'index',
                    '_index':index,
                    '_id': str( results.iloc[i][f'source_{source_id}'] + "_" + results.iloc[i][f'sink_{sink_id}']),
                    str(f'_{index}'):{
                        f'source_{source_id}':results.iloc[i][f'source_{source_id}'],
                        f'sink_{sink_id}':results.iloc[i][f'sink_{sink_id}'],
                        'co2_transported':results.iloc[i]['co2_transported']
                    }
                }
            )

    # Export
    success, fail = helpers.bulk(es, actions_network)

    return success, fail


# Import network
def network_import(host, auth, index, query):

    # Host setup
    es = Elasticsearch(hosts=host)
    es.indices.refresh(index=index)

    # Search
    response = es.search(index=index, body=query)
    documents = response['hits']['hits']

    # Create df results
    list_docs = []
    for doc in documents:
        list_docs.append(doc['_source'][f'_{index}'])

    df_network = pd.DataFrame(list_docs)

    return df_network


# Create network map
def network_map(network, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon):

    source_id = f'source_{source_id}'
    sink_id = f'sink_{sink_id}'

    network[source_lat] = pd.merge(network, source, on=source_id, how="inner")[source_lat]
    network[source_lon] = pd.merge(network, source, on=source_id, how="inner")[source_lon]

    # network[sink_lat] = pd.merge(network, df_sink, on=sink_id, how="inner")[sink_lat]
    # network[sink_lon] = pd.merge(network, df_sink, on=sink_id, how="inner")[sink_lon]

    # breakpoint()
    network[sink_lat] = pd.Series()
    network[sink_lon] = pd.Series()

    for i, row in network.iterrows():
        if row[sink_id] == "Atmosphere":
            network.at[i,sink_lat] = row[source_lat]
            network.at[i,sink_lon] = row[source_lon]
        else:
            network.at[i,sink_lat] = sink[sink[sink_id] == int(float(row[sink_id]))][sink_lat]
            network.at[i,sink_lon] = sink[sink[sink_id] == int(float(row[sink_id]))][sink_lon]

    # df_concat = pd.concat([network[sink_lat], df_sink[sink_lat]], axis=1, join='inner')
    
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

    return map


# Step . Results calculation
def results_calc(results, source_id, sink_id, emission_cost, capture_cost):

    quantity_stored = 0 
    quantity_emitted = 0
    expense_capture = 0 
    expense_tax = 0
    source_id = str(f'source_{source_id}')
    sink_id = str(f'sink_{sink_id}')


    for i, row in results.iterrows():
        if row[sink_id] == 'Atmosphere':
            quantity_emitted += row['co2_transported']
            expense_tax += row['co2_transported'] * emission_cost
        else:
            quantity_stored += row['co2_transported']
            expense_capture += row['co2_transported'] * capture_cost

    res = {
        'quantity_emitted':quantity_emitted,
        'quantity_stored':quantity_stored,
        'expense_tax':expense_tax,
        'expense_capture':expense_capture
    }
    res = pd.DataFrame(res, index=[1])
    
    return res