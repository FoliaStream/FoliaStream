import os
import errno
import requests
import statistics
import folium
import branca
import pulp
import ast
import shutil

import pandas as pd
import numpy as np

from loguru import logger
from geopy.distance import geodesic
from pipe.functions.functions_II import request_url, distance_matrix, cost_matrix, elbow_method, create_clusters, distance_matrix_II, cost_matrix_source_centr, cost_matrix_centr_sink, create_fully_connected_graph, generate_all_paths, path_based_mcf_model, visualize_flow_map


# //////////////////////////////////////////////////////
#                  FUNCTIONS I LEVEL
# //////////////////////////////////////////////////////

#----------------------
# STEP . Clean folder
#----------------------

def clean_folder(path: str):

    # Check if exists
    if os.path.exists(path):
        # Delete
        shutil.rmtree(path)
    else:
        pass

    return path


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


#----------------------
# STEP . Source load 
#----------------------

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




#----------------------
# STEP . Sink load
#----------------------

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

#-----------------
# STEP . Nodes map
#-----------------

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


#----------------------
# STEP . Create matrix
#----------------------

def create_matrix(source, sink, source_id, source_lat, source_lon, sink_id, sink_lat, sink_lon, emission_cost, capture_cost, url, transport_cost, transport_method):

    # Distance
    matrix_distance = distance_matrix(url, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, transport_method)

    # Cost
    matrix_cost = cost_matrix(matrix_distance, transport_method, transport_cost, emission_cost, capture_cost)

    return matrix_cost






#----------------------------
# STEP . Network optimization
#----------------------------

def network_optimization_levelized(df_source, df_sink, df_cost_matrix, source_id, sink_id, source_capacity, sink_capacity, emission_cost, transport_method, quantity_cost_segments):

    df_cost_matrix = df_cost_matrix.rename(columns={"Unnamed: 0":sink_id})
    
    # Network initialization
    network = pulp.LpProblem("Network_problem", pulp.LpMinimize)

    # Generate nodes
    source_list = df_source[source_id].astype(str)
    sink_list = list(df_sink[sink_id].astype(str))
    sink_list.append("Atmosphere")  

    nodes = []
    for i in source_list:
        nodes.append(i)
    for i in sink_list:
        nodes.append(i)
    
    # Generate arcs
    arcs = []
    arc_capacities = {}
    for i in source_list:
        for j in sink_list:
            if j != "Atmosphere":
                arcs.append((f"source_id_{i}", f"sink_id_{j}"))
                arc_capacities[(f"source_id_{i}", f"sink_id_{j}")] = 10000000 # 10Mty capacity of pipes
        arcs.append((f"source_id_{i}","Atmosphere"))
        arc_capacities[(f"source_id_{i}","Atmosphere")] = 99999999999999999999999 # infinite value for emitting

    
    # Flow variables
    flow_vars = pulp.LpVariable.dicts("Flow", arcs, 0, None, pulp.LpContinuous)

    
    # Cost functions
    for method, data in quantity_cost_segments.items():
        if type(list(quantity_cost_segments[method].keys())[0]) == str:
            quantity_cost_segments[method] = {ast.literal_eval(k): v for k, v in quantity_cost_segments[method].items()}
        else:
            pass

    # Set sink_id as index of transport cost matrix
    transport_cost = df_cost_matrix.set_index(sink_id)

    cost_segments = {}
    for i in source_list:
        for j in sink_list:
            cost_segments[(f"source_id_{i}",f"sink_id_{j}")] = [
                (0,1000,quantity_cost_segments[transport_method][(0, 1000)]*transport_cost.at[j,i]),
                (1000,5000,quantity_cost_segments[transport_method][(1000,5000)]*transport_cost.at[j,i]),
                (5000,10000,quantity_cost_segments[transport_method][(5000,10000)]*transport_cost.at[j,i]),
                (10000,20000,quantity_cost_segments[transport_method][(10000,20000)]*transport_cost.at[j,i]),
                (20000,30000,quantity_cost_segments[transport_method][(20000,30000)]*transport_cost.at[j,i]),
                (30000,50000,quantity_cost_segments[transport_method][(30000,50000)]*transport_cost.at[j,i]),
                (50000,100000,quantity_cost_segments[transport_method][(50000,100000)]*transport_cost.at[j,i]),
                (100000,1000000000000,quantity_cost_segments[transport_method][(100000,1000000000000)]*transport_cost.at[j,i])
            ]
        cost_segments[(f"source_id_{i}",f"Atmosphere")] = [(0,1000000000000, emission_cost)]
    

    # Segment variables
    segment_vars = {}
    for arc in arcs:
        segment_vars[arc] = []
        for i, (start, end, slope) in enumerate(cost_segments[arc]):
            var = pulp.LpVariable(f"Segment_{arc}_{i}", 0, end - start, pulp.LpContinuous)
            segment_vars[arc].append((var, start, end, slope))
        
    # Objective function
    network += pulp.lpSum([var * slope for arc in arcs for var, start, end, slope in segment_vars[arc]]), "TotalCost"

    # Constraints
    for i, rsource in df_source.iterrows():
        network += sum(flow_vars[(f"source_id_{rsource[source_id]}", f"sink_id_{rsink[sink_id]}")] for j, rsink in df_sink.iterrows() if rsink[sink_capacity] != "Atmosphere") + flow_vars[(f"source_id_{rsource[source_id]}", "Atmosphere")] == rsource[source_capacity], f"source_id_{rsource[source_id]}_outflow"
    
    for i, rsink in df_sink.iterrows():
            network += sum(flow_vars[(f"source_id_{rsource[source_id]}", f"sink_id_{rsink[sink_id]}")] for j, rsource in df_source.iterrows()) <= rsink[sink_capacity], f"sink_id_{rsink[sink_id]}_inflow"

    for arc in arcs:
        network += flow_vars[arc] <= arc_capacities[arc], f"Capacity_{arc}"

    for arc in arcs:
        network += flow_vars[arc] == pulp.lpSum([var for var, start, end, slope in segment_vars[arc]]), f"Piecewise_{arc}"
    
    # Solution
    network.solve()

    # Check status
    status = pulp.LpStatus[network.status]
    print(status)

    # if status == "Infeasible":
    #     print("Infeasible model")
    # else:
    #     for arc in arcs:
    #         print(f"Flow on arc {arc}: {flow_vars[arc].varValue}")
    #     print("Total Cost: ", pulp.value(network.objective))

    # Export results
    results = []

    for arc in arcs:
        if flow_vars[arc].varValue > 0:
            if arc[1] != "Atmosphere":
                results.append({
                    'source_id':arc[0][10:],
                    'sink_id':arc[1][8:],
                    'co2_transported':flow_vars[arc].varValue
                })
            else:
                results.append({
                    'source_id':arc[0][10:],
                    'sink_id':arc[1],
                    'co2_transported':flow_vars[arc].varValue
                })
            
    df_results = pd.DataFrame(results)
    
    return df_results





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


def network_optimization_klust(df_source, df_sink, df_cost_matrix, source_id, sink_id, source_capacity, sink_capacity, url, transport_method, transport_cost, emission_cost, capture_cost):
    
    # Run optimization for all nodes
    mcf_I_results = network_optimization(df_source, df_sink, df_cost_matrix, source_id, sink_id, source_capacity, sink_capacity)

    # Extract nodes failed to connect
    unconnected_df = mcf_I_results[mcf_I_results['sink_id'] == "Atmosphere"]

    # Check if clusterization needed 
    if len(unconnected_df) == 0:
        # No need clustering
        return mcf_I_results
    else:
        # Need clustering
        df_source[source_id] = pd.Series(df_source[source_id].astype(str))
        df_sink[sink_id] = pd.Series(df_sink[sink_id].astype(str))

        df_source = df_source.rename(columns={source_id:f'source_{source_id}'})
        df_sink = df_sink.rename(columns={sink_id:f'sink_{sink_id}'})

        unconnected_df = pd.merge(unconnected_df, df_source, how='left', on='source_id')   
        unconnected_df = unconnected_df.rename(columns={'lat':'source_lat', 'lon':'source_lon'})    

        # Cluster the unconnected nodes
        el_point = elbow_method(unconnected_df, 'source_lat', 'source_lon')
        unconnected_df = unconnected_df.reset_index(drop=True)
        centr, labs = create_clusters(unconnected_df, 'source_lat', 'source_lon', el_point, 'co2_transported')

        # Merge centroids and unconndected points
        unconnected_df['cluster_lab'] = pd.Series(labs)
        centr = centr.rename(columns={'id':'cluster_lab','source_lat':'cluster_lat','source_lon':'cluster_lon'})
        unconnected_df = pd.merge(unconnected_df, centr, on='cluster_lab', how='left')

        # New distance and cost matrix

        # Count number of nodes per cluster
        cluster_size = unconnected_df.groupby('cluster_lab').size()
        # Distance source - centroid
        matrix_dist_source_centr = distance_matrix_II(url, unconnected_df, centr, 'source_id', 'cluster_lab', "source_lat", "cluster_lat", "source_lon", "cluster_lon", transport_method)
        # Cost source - centroid (individual)
        matrix_cost_source_centr = cost_matrix_source_centr(matrix_dist_source_centr, transport_method, transport_cost, capture_cost)
        # Distance centroid - sink
        matrix_dist_centr_sink = distance_matrix_II(url, centr, df_sink, 'cluster_lab', 'sink_id', 'cluster_lat', 'latitude', 'cluster_lon', 'longitude', transport_method)
        # Cost centroid - sink (shared)
        matrix_cost_centr_sink = cost_matrix_centr_sink(matrix_dist_centr_sink, transport_method, transport_cost, cluster_size)

        # General cost matrix setup
        matrix = pd.DataFrame(index=df_sink['sink_id'], columns=unconnected_df['source_id'])
        for i in matrix.index: #sinks
            for j in matrix.columns: #sources
                matrix.at[i,j] = matrix_cost_source_centr.loc[unconnected_df.set_index('source_id')['cluster_lab'].loc[j], j]
                matrix.at[i,j] = matrix.at[i,j] + matrix_cost_centr_sink.loc[i, unconnected_df.set_index('source_id')['cluster_lab'].loc[j]]
        atmosphere_row = pd.DataFrame({col:emission_cost for col in matrix.columns}, index=["Atmosphere"])
        matrix = pd.concat([matrix, atmosphere_row], axis=0)
        matrix = matrix.reset_index().rename(columns={'index':'Unnamed: 0'})

        # Run optimization for clustered nodes ()
        mcf_II_results = network_optimization(unconnected_df, df_sink, matrix, "source_id", "sink_id", source_capacity, sink_capacity)


        # Merge results
        mcf_II_results = mcf_II_results.merge(unconnected_df[['source_id', 'cluster_lab', 'cluster_lat', 'cluster_lon']], on='source_id', how='left')

        # results = mcf_I_results.merge(mcf_II_results[['source_id','sink_id','cluster_lab','cluster_lat','cluster_lon']],
        #                               on='source_id',
        #                               how='left',
        #                               suffixes=('_I','_II'))
        
        # results['sink_id'] = results['sink_id_II'].combine_first(results['sink_id_I'])
        # cluster_update = ~results['sink_id_II'].isna()
        # results['cluster_lab'] = results['cluster_lab'].where(cluster_update, None)
        # results['cluster_lat'] = results['cluster_lat'].where(cluster_update, None)
        # results['cluster_lon'] = results['cluster_lon'].where(cluster_update, None)

        # results = results.drop(columns=['sink_id_I', 'sink_id_II'])

        merged = mcf_I_results.merge(mcf_II_results[['source_id', 'sink_id']], 
                  on='source_id', 
                  how='left',
                  suffixes=('_original', '_updated'))

        is_updated = (merged['sink_id_original'] == 'Atmosphere') & (merged['sink_id_updated'] != 'Atmosphere')

        merged = merged.merge(mcf_II_results[['source_id', 'cluster_lab', 'cluster_lat', 'cluster_lon']], 
                     on='source_id', 
                     how='left')
        
        results = mcf_I_results.copy()
        results['sink_id'] = merged['sink_id_updated'].combine_first(merged['sink_id_original'])
        results['cluster_lab'] = merged['cluster_lab'].where(is_updated, np.nan)
        results['cluster_lat'] = merged['cluster_lat'].where(is_updated, np.nan)
        results['cluster_lon'] = merged['cluster_lon'].where(is_updated, np.nan)

        return results





def network_optimization_klust_levelized(df_source, df_sink, df_cost_matrix, source_id, sink_id, source_capacity, sink_capacity, url, transport_method, transport_cost, emission_cost, capture_cost, quantity_cost_segments):

    # Run optimization for all nodes
    mcf_I_results = network_optimization_levelized(df_source, df_sink, df_cost_matrix, source_id, sink_id, source_capacity, sink_capacity, emission_cost, transport_method, quantity_cost_segments)

    # Extract nodes failed to connect
    unconnected_df = mcf_I_results[mcf_I_results['sink_id'] == "Atmosphere"]

    # Check if clusterization needed 
    if len(unconnected_df) == 0:
        # No need clustering
        return mcf_I_results
    else:
        # Need clustering
        df_source[source_id] = pd.Series(df_source[source_id].astype(str))
        df_sink[sink_id] = pd.Series(df_sink[sink_id].astype(str))

        df_source = df_source.rename(columns={source_id:f'source_{source_id}'})
        df_sink = df_sink.rename(columns={sink_id:f'sink_{sink_id}'})

        unconnected_df = pd.merge(unconnected_df, df_source, how='left', on='source_id')   
        unconnected_df = unconnected_df.rename(columns={'lat':'source_lat', 'lon':'source_lon'})    

        # Cluster the unconnected nodes
        el_point = elbow_method(unconnected_df, 'source_lat', 'source_lon')
        unconnected_df = unconnected_df.reset_index(drop=True)
        centr, labs = create_clusters(unconnected_df, 'source_lat', 'source_lon', el_point, 'co2_transported')

        # Merge centroids and unconndected points
        unconnected_df['cluster_lab'] = pd.Series(labs)
        centr = centr.rename(columns={'id':'cluster_lab','source_lat':'cluster_lat','source_lon':'cluster_lon'})
        unconnected_df = pd.merge(unconnected_df, centr, on='cluster_lab', how='left')

        # New distance and cost matrix

        # Count number of nodes per cluster
        cluster_size = unconnected_df.groupby('cluster_lab').size()
        # Distance source - centroid
        matrix_dist_source_centr = distance_matrix_II(url, unconnected_df, centr, 'source_id', 'cluster_lab', "source_lat", "cluster_lat", "source_lon", "cluster_lon", transport_method)
        # Cost source - centroid (individual)
        matrix_cost_source_centr = cost_matrix_source_centr(matrix_dist_source_centr, transport_method, transport_cost, capture_cost)
        # Distance centroid - sink
        matrix_dist_centr_sink = distance_matrix_II(url, centr, df_sink, 'cluster_lab', 'sink_id', 'cluster_lat', 'latitude', 'cluster_lon', 'longitude', transport_method)
        # Cost centroid - sink (shared)
        matrix_cost_centr_sink = cost_matrix_centr_sink(matrix_dist_centr_sink, transport_method, transport_cost, cluster_size)

        # General cost matrix setup
        matrix = pd.DataFrame(index=df_sink['sink_id'], columns=unconnected_df['source_id'])
        for i in matrix.index: #sinks
            for j in matrix.columns: #sources
                matrix.at[i,j] = matrix_cost_source_centr.loc[unconnected_df.set_index('source_id')['cluster_lab'].loc[j], j]
                matrix.at[i,j] = matrix.at[i,j] + matrix_cost_centr_sink.loc[i, unconnected_df.set_index('source_id')['cluster_lab'].loc[j]]
        atmosphere_row = pd.DataFrame({col:emission_cost for col in matrix.columns}, index=["Atmosphere"])
        matrix = pd.concat([matrix, atmosphere_row], axis=0)
        matrix = matrix.reset_index().rename(columns={'index':'Unnamed: 0'})

        # Run optimization for clustered nodes ()
        mcf_II_results = network_optimization_levelized(unconnected_df, df_sink, matrix, "source_id", "sink_id", source_capacity, sink_capacity, emission_cost, transport_method, quantity_cost_segments)


        # Merge results
        mcf_II_results = mcf_II_results.merge(unconnected_df[['source_id', 'cluster_lab', 'cluster_lat', 'cluster_lon']], on='source_id', how='left')

        # results = mcf_I_results.merge(mcf_II_results[['source_id','sink_id','cluster_lab','cluster_lat','cluster_lon']],
        #                               on='source_id',
        #                               how='left',
        #                               suffixes=('_I','_II'))
        
        # results['sink_id'] = results['sink_id_II'].combine_first(results['sink_id_I'])
        # cluster_update = ~results['sink_id_II'].isna()
        # results['cluster_lab'] = results['cluster_lab'].where(cluster_update, None)
        # results['cluster_lat'] = results['cluster_lat'].where(cluster_update, None)
        # results['cluster_lon'] = results['cluster_lon'].where(cluster_update, None)

        # results = results.drop(columns=['sink_id_I', 'sink_id_II'])

        merged = mcf_I_results.merge(mcf_II_results[['source_id', 'sink_id']], 
                  on='source_id', 
                  how='left',
                  suffixes=('_original', '_updated'))

        is_updated = (merged['sink_id_original'] == 'Atmosphere') & (merged['sink_id_updated'] != 'Atmosphere')

        merged = merged.merge(mcf_II_results[['source_id', 'cluster_lab', 'cluster_lat', 'cluster_lon']], 
                     on='source_id', 
                     how='left')
        
        results = mcf_I_results.copy()
        results['sink_id'] = merged['sink_id_updated'].combine_first(merged['sink_id_original'])
        results['cluster_lab'] = merged['cluster_lab'].where(is_updated, np.nan)
        results['cluster_lat'] = merged['cluster_lat'].where(is_updated, np.nan)
        results['cluster_lon'] = merged['cluster_lon'].where(is_updated, np.nan)

        return results


def network_optimization_dijkstra(df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, emission_cost):

    df_source[source_id] = df_source[source_id].astype(int)
    df_sink[sink_id] = df_sink[sink_id].astype(int)

    df_source = df_source.set_index(df_source[source_id])
    df_sink = df_sink.set_index(df_sink[sink_id])

    # Create graph
    graph = create_fully_connected_graph(df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon)

    # Generate all paths
    path_registry = generate_all_paths(df_source, df_sink, graph, source_id, sink_id)

    # Create and solve the MCF model
    prob, path_vars, atmo_vars = path_based_mcf_model(df_source, df_sink, path_registry, emission_cost, source_id, sink_id)
    prob.solve()

    # results??

    # Check solution status
    status = pulp.LpStatus[prob.status]
    print("Status: ", status)

    if status == 'Infeasible':
        print("The model is infeasible.")
    else:
        # Print results
        print("\n Flow Results:")
        for (i,j) in path_registry:
            flow = path_vars[(i,j)].varValue
            if flow > 0:
                path = path_registry[(i,j)]['path']
                print(f"Flow from source_{i} to sink_{j}: {flow:.2f} tons")
                print(f"Path: {'==>'.join(path)}")
                print(f"Distance: {path_registry[(i,j)]['actual_distance']:.2f} km")


        # Print atmospheric emissions
        for i, row in df_source.iterrows():
            flow = atmo_vars[i].varValue
            if flow>0:
                print(f"Emissions from source_{i} to atmosphere: {flow:.2f} units")
        print("\n Total Cost:", pulp.value(prob.objective))
    
    results = []

    for (i,j) in path_registry:
        flow = path_vars[(i,j)].varValue
        if flow > 0:
            results.append({
                'source_id':i,
                'sink_id':j,
                'co2_transported':flow,
                'dijkstra_path': path_registry[(i,j)]['path']
            })
    for i, row in df_source.iterrows():
        flow = atmo_vars[i].varValue
        if flow > 0:
            results.append({
                'source_id':i,
                'sink_id':'Atmosphere',
                'co2_transported':flow,
                'dijkstra_path':[f'source_{i}','Atmosphere']
            })

    df_results = pd.DataFrame(results)
    df_path_vars = pd.DataFrame.from_dict(path_vars, orient='index', columns=['flow_name'])
    df_path_vars = df_path_vars.reset_index()
    df_path_registry = pd.DataFrame(path_registry)

    return df_results, df_path_registry, df_path_vars





#-------------------
# STEP . Network map
#-------------------
 
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



def network_map_klust(network, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon):
    
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

    map_lat = statistics.mean([sink[sink_lat].mean(),source[source_lat].mean()])
    map_lon = statistics.mean([sink[sink_lon].mean(),source[source_lon].mean()])
    map_coords = (map_lat, map_lon)
    map = folium.Map(map_coords, zoom_start=4)


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

    if 'cluster_lat' in network.columns:

        for i,row in network.iterrows():
            if np.isnan(row['cluster_lat']):
                pass
            else:
                folium.Marker(
                    location=[row['cluster_lat'], row['cluster_lon']],
                    popup=str(row['cluster_lab']),
                    icon = folium.Icon(color='green', icon="")
            ).add_to(map)

        network_noklust = network[np.isnan(network['cluster_lab'])]
        network_klust = network[~np.isnan(network['cluster_lab'])]

        direct_links = []
        for i, row in network_noklust.iterrows():
            direct_links.append([[row[source_lat], row[source_lon]], [row[sink_lat], row[sink_lon]], str(f"{row[source_id]}__{row[sink_id]}")])
        for i in range(len(direct_links)):
            line = folium.PolyLine(locations=[direct_links[i][0], direct_links[i][1]], popup=direct_links[i][2])
            line.add_to(map)
        
        indirect_links = []
        for i, row in network_klust.iterrows():
            if row[sink_id] == 'Atmosphere':
                indirect_links.append([[row[source_lat], row[source_lon]], [row[source_lat], row[source_lon]], str(f"{row[source_id]}__{row[sink_id]}")])
            else:
                indirect_links.append([[row[source_lat], row[source_lon]], [row['cluster_lat'], row['cluster_lon']], str(f"{row[source_id]}__{row['cluster_lab']}")])
                indirect_links.append([[float(row['cluster_lat']), float(row['cluster_lon'])], [float(row[sink_lat]), float(row[sink_lon])], str(f"{row['cluster_lab']}__{row[sink_id]}")])

        for i in range(len(indirect_links)):
            line = folium.PolyLine(locations=[indirect_links[i][0], indirect_links[i][1]], popup=indirect_links[i][2])
            line.add_to(map)

    else:
        direct_links = []
        for i, row in network.iterrows():
            direct_links.append([[row[source_lat], row[source_lon]], [row[sink_lat], row[sink_lon]], str(f"{row[source_id]}__{row[sink_id]}")])
        for i in range(len(direct_links)):
            line = folium.PolyLine(locations=[direct_links[i][0], direct_links[i][1]], popup=direct_links[i][2])
            line.add_to(map)


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
        &nbsp; Cluster node &nbsp; <i class="fa fa-circle" style="color:green"></i><br>
    </div>
    {% endmacro %}
    '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)
    map.get_root().add_child(legend)
    
    return map



def network_map_dijkstra(network, df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, path_registry, path_vars):

    df_source[source_id] = df_source[source_id].astype(int)
    df_sink[sink_id] = df_sink[sink_id].astype(int)  
    df_source = df_source.set_index(df_source[source_id])
    df_sink = df_sink.set_index(df_sink[sink_id])

    lat_max = max(max(df_sink[sink_lat]), max(df_source[source_lat]))
    lat_min = min(min(df_sink[sink_lat]), min(df_source[source_lat]))
    lon_max = max(max(df_sink[sink_lon]), max(df_source[source_lon]))
    lon_min = min(min(df_sink[sink_lon]), min(df_source[source_lon]))

    map = folium.Map(location=((lat_max + lat_min)/2, (lon_max + lon_min)/2), zoom_start=6)

    
    # Add markers for sources and sinks

    for i, row in network.iterrows():
        folium.Marker(
            location = (df_source.loc[int(row["source_id"])][source_lat],df_source.loc[int(row["source_id"])][source_lon]),
            icon=folium.Icon(color='red'),
            tooltip = f"source_{row['source_id']}"
        ).add_to(map)

    for i, row in network.iterrows():
        if row['sink_id'] != 'Atmosphere':
            folium.Marker(
                location = (df_sink.loc[int(row["sink_id"])][sink_lat],df_sink.loc[int(row["sink_id"])][sink_lon]),
                icon = folium.Icon(color='blue'),
                tooltip=f"sink_{row['sink_id']}"
            ).add_to(map)
    
    dijkstra_map = visualize_flow_map(map, network, df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon)

    return dijkstra_map