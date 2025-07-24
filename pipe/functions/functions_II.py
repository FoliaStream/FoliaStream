from requests.models import PreparedRequest
from sklearn.cluster import KMeans
from kneed import KneeLocator 
from geopy.distance import geodesic
from kneed import KneeLocator 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import networkx as nx
import pandas as pd
import numpy as np

import requests
import pulp
import folium
import ast
import heapq

# ////////////////////////////////////////////////
#                   FUNCTIONS II
# ////////////////////////////////////////////////

# Generate url for API query 
def request_url(url, params):
    
    request = PreparedRequest()
    request.prepare_url(url, params)

    return request.url


# Distance matrix

def distance_matrix(url, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, transport_method, batch_size):

    matrix = pd.DataFrame()

    if transport_method == 'pipe':

        for i, rsink in sink.iterrows():
            for j, rsource in source.iterrows():
                matrix.at[i,j] = geodesic((float(rsource[source_lat]),float(rsource[source_lon])), (float(rsink[sink_lat]),float(rsink[sink_lon]))).km


    elif transport_method == 'truck_ship':

        ####### Hitting the limit dimension for osrm api -- batching the coordinates to reduce request size and combine multiple requests #######
        # sink = sink.rename(columns={'latitude':'lat', 'longitude':'lon'})
        # combined_df = pd.concat([source, sink], ignore_index=True)
        # all_coords  = ";".join([f"{lon},{lat}" for lat, lon in zip(combined_df['lat'].round(4), combined_df['lon'].round(4))])

        # params = {
        #     "sources": ";".join(map(str, range(len(source)))),
        #     "destinations": ";".join(map(str, range(len(source), len(source)+len(sink)))),
        #     "annotations": "distance"  # Request both
        # }

        # # Make the request
        # response = requests.get(url + all_coords, params=params)
        # breakpoint()
        # data = response.json()

        # # Truck matrix

        # matrix = pd.DataFrame(data['distances'])

        # # Add ship transportation (distance from final point)
        # for col in range(len(matrix.columns)):
        #     matrix[col] = matrix[col] + data['destinations'][0]['distance']

        # # Transpose and convert to km
        # matrix = matrix.T/1000
        # Standardize column names

        # batch_size = 250
        sink = sink.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        combined_df = pd.concat([source, sink], ignore_index=True)
        
        # Initialize empty matrix
        matrix = np.zeros((len(source), len(sink)))
        
        # Process in batches
        for i in range(0, len(source), batch_size):
            source_batch = source.iloc[i:i+batch_size]
            source_indices = list(range(i, min(i+batch_size, len(source))))
            
            for j in range(0, len(sink), batch_size):
                sink_batch = sink.iloc[j:j+batch_size]
                sink_indices = list(range(len(source) + j, 
                                        len(source) + min(j+batch_size, len(sink))))
                
                # Prepare coordinates
                batch_coords = pd.concat([source_batch, sink_batch])
                coords_str = ";".join([f"{lon:.4f},{lat:.4f}" 
                                    for lat, lon in zip(batch_coords['lat'], batch_coords['lon'])])
                
                # Prepare parameters
                params = {
                    "sources": ";".join(map(str, range(len(source_batch)))),
                    "destinations": ";".join(map(str, range(len(source_batch), 
                                            len(source_batch)+len(sink_batch)))),
                    "annotations": "distance"
                }
                
                # Make request
                response = requests.get(url + coords_str, params=params)

                data = response.json()
                
                # Fill matrix
                batch_matrix = np.array(data['distances'])
                matrix[i:i+batch_size, j:j+batch_size] = batch_matrix
        
        # Convert to DataFrame and km
        matrix = pd.DataFrame(matrix).T/1000





    else:
        raise Exception("Error: Wrong transportation method")

    matrix = matrix.set_index(sink[sink_id])
    matrix = matrix.rename(columns=source[source_id])

    return matrix




# Cost matrix (careful, the transport cost is not dependent on the quantity transported --> multiply this cost by the total number of tons of relative source // adjust optimization model to consider transport cost dependent on the number of tons transported, more correct)
def cost_matrix(matrix, method, transport_cost, emission_cost, capture_cost):

    for col in matrix.columns:

        for id in matrix[col].index:
            

            if matrix.at[id,col]<180:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['less_180'])

            elif matrix.at[id,col] >= 180 and matrix.at[id,col] < 500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_180_500'])
            
            elif matrix.at[id,col] >= 500 and matrix.at[id,col] < 750:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_500_750'])

            elif matrix.at[id,col] >= 750 and matrix.at[id,col] < 1500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_750_1500'])

            else:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['more_1500'])
            
    atmosphere_row = pd.DataFrame({col:emission_cost for col in matrix.columns}, index=["Atmosphere"])

    matrix = pd.concat([matrix, atmosphere_row], axis=0)

    return matrix



# Elbow method 

# Calculate optimal number of clusters using the elbow method
def elbow_method(data, x_ax, y_ax):

    coords = []
    for i,row in data.iterrows():
        coords.append([row[x_ax], row[y_ax]])
    
    wcss = []
    max_clusters = len(data)
    for k in range(1,max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(coords)
        wcss.append(kmeans.inertia_)
    

    knee_locator = KneeLocator(range(1, max_clusters + 1), wcss, curve="convex", direction="decreasing")
    elbow_point = knee_locator.elbow
    
    # Minimum one cluster
    if elbow_point == None:
        elbow_point = 1

    return elbow_point



# Create clusters

# Cluster data using weighted measure
def create_clusters(data, x_ax, y_ax, k, weight):

    coord_x = np.zeros(len(data))
    coord_y = np.zeros(len(data))

    for i, row in data.iterrows():
        coord_x[i] = data[x_ax][i]
        coord_y[i] = data[y_ax][i]
        
    X = np.array(list(zip(coord_x,coord_y))).reshape(len(coord_x),2)
    scaled_weights = (data[weight]/1000).astype(int)
    X_weighted = np.repeat(X, scaled_weights, axis=0)

    gmm = GaussianMixture(n_components=k).fit(X_weighted)

    centroids = gmm.means_.tolist()
    centroids = pd.DataFrame(centroids, columns=[x_ax, y_ax])
    centroids['id'] = pd.Series(centroids.index)
    labels = gmm.predict(X).tolist()


    return centroids, labels




# Distance matrix MCFII
def distance_matrix_II(url, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, transport_method):

    matrix = pd.DataFrame()

    if transport_method == 'pipe':

        for i, rsink in sink.iterrows():
            for j, rsource in source.iterrows():
                matrix.at[i,j] = geodesic((float(rsource[source_lat]),float(rsource[source_lon])), (float(rsink[sink_lat]),float(rsink[sink_lon]))).km


    elif transport_method == 'truck_ship':

        sink = sink.rename(columns={sink_lat:source_lat, sink_lon:source_lon})
        combined_df = pd.concat([source, sink], ignore_index=True)
        all_coords  = ";".join([f"{lon},{lat}" for lat, lon in zip(combined_df[source_lat], combined_df[source_lon])])

        params = {
            "sources": ";".join(map(str, range(len(source)))),
            "destinations": ";".join(map(str, range(len(source), len(source)+len(sink)))),
            "annotations": "distance"  # Request both
        }

        # Make the request
        response = requests.get(url + all_coords, params=params)
        data = response.json()

        # Truck matrix
        matrix = pd.DataFrame(data['distances'])

        # Add ship transportation (distance from final point)
        for col in range(len(matrix.columns)):
            matrix[col] = matrix[col] + data['destinations'][0]['distance']

        # Transpose and convert to km
        matrix = matrix.T/1000

    else:
        raise Exception("Error: Wrong transportation method")

    matrix = matrix.set_index(sink[sink_id])
    matrix = matrix.rename(columns=source[source_id])

    return matrix



# Cost matrix source-centr
def cost_matrix_source_centr(matrix, method, transport_cost, capture_cost):

    for col in matrix.columns:

        for id in matrix[col].index:

            if matrix.at[id,col]<180:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['less_180'])

            elif matrix.at[id,col] >= 180 and matrix.at[id,col] < 500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_180_500'])
            
            elif matrix.at[id,col] >= 500 and matrix.at[id,col] < 750:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_500_750'])

            elif matrix.at[id,col] >= 750 and matrix.at[id,col] < 1500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_750_1500'])

            else:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['more_1500'])
            
    return matrix





# Cost matrix centr-sink
def cost_matrix_centr_sink(matrix, method, transport_cost, cluster_size):

    for col in matrix.columns:

        for id in matrix[col].index:

            if matrix.at[id,col]<180:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['less_180'])/cluster_size[col]

            elif matrix.at[id,col] >= 180 and matrix.at[id,col] < 500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_180_500'])/cluster_size[col]
            
            elif matrix.at[id,col] >= 500 and matrix.at[id,col] < 750:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_500_750'])/cluster_size[col]

            elif matrix.at[id,col] >= 750 and matrix.at[id,col] < 1500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_750_1500'])/cluster_size[col]

            else:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['more_1500'])/cluster_size[col]

    return matrix


# DIJKSTRA FUNCTIONS

# Fully connected graph 
def create_fully_connected_graph(df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon):

    df_source = df_source.set_index(df_source[source_id])
    df_sink = df_sink.set_index(df_sink[sink_id])

    # Init graph
    G = nx.DiGraph()

    # Add source nodes
    for idx, row in df_source.iterrows():
        node_id = f"source_{idx}"
        G.add_node(node_id,
                   pos=(row[source_lat], row[source_lon]),
                   type='source')
    
    # Add sink nodes
    for idx, row in df_sink.iterrows():
        node_id = f"sink_{idx}"
        G.add_node(node_id,
                   pos=(row[sink_lon], row[sink_lat]),
                   type='sink')
        
    # Create fully connected graph with geodesic distances 
    for i, (i_id, i_data) in enumerate(G.nodes(data=True)):
        i_lat = df_source.loc[int(i_id.split('_')[1])][source_lat] if 'source' in i_id else df_sink.loc[int(i_id.split('_')[1])][sink_lat]
        i_lon = df_source.loc[int(i_id.split('_')[1])][source_lon] if 'source' in i_id else df_sink.loc[int(i_id.split('_')[1])][sink_lon]

        for j,(j_id, j_data) in enumerate(G.nodes(data=True)):
            if i_id != j_id: # Avoid self loops
                j_pos = j_data['pos']
                j_lat = df_source.loc[int(j_id.split('_')[1])][source_lat] if 'source' in j_id else df_sink.loc[int(j_id.split('_')[1])][sink_lat]
                j_lon = df_source.loc[int(j_id.split('_')[1])][source_lon] if 'source' in j_id else df_sink.loc[int(j_id.split('_')[1])][sink_lon]

                # Calculate geodesic distance
                weight = geodesic((i_lat, i_lon), (j_lat, j_lon)).kilometers
                G.add_edge(i_id, j_id, weight=weight)

    return G


# Determine Dijkstra optimal paths
def path_dependent_dijkstra(G, source, target):

    # Track distance, paths, and hop counts
    distances = {node: float('infinity') for node in G.nodes()}
    distances[source] = 0
    previous_nodes = {node: None for node in G.nodes()}
    hop_counts = {node:0 for node in G.nodes()}
    hop_counts[source] = 1 # start with 1 node in the path

    # Priority queue for nodes to visti: (adjusted_distance, node)
    priority_queue = [(0,source)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Stop if reach target
        if current_node == target:
            break
    
        # Skip if already found better path
        if current_distance > distances[current_node]:
            continue

        # Check all neighbors
        for neighbor in G.neighbors(current_node):

            # Raw edge weight
            edge_weight = G[current_node][neighbor]['weight']

            # Adjust weight based on hop count (divide by hop count)
            adjusted_weight = edge_weight / hop_counts[current_node]
            # TODO filter out nodes that are not transporting --> iterative approach 

            # Calculate new distances
            distance = distances[current_node] + adjusted_weight

            # Update if better path found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                hop_counts[neighbor] = hop_counts[current_node] + 1
                heapq.heappush(priority_queue, (distance, neighbor))

    # Reconstruct path
    path = []
    current_node = target
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    # Calculate actual distance (not adjusted) for the found path 
    actual_distance = 0 
    for i in range(len(path)-1):
        actual_distance += G[path[i]][path[i+1]]['weight']
    
    return path, distances[target], actual_distance, hop_counts[target]




# Generate Dijkstra paths
def generate_all_paths(df_source, df_sink, graph, source_id, sink_id):

    df_source = df_source.set_index(df_source[source_id])
    df_sink = df_sink.set_index(df_sink[sink_id])

    path_registry = {}

    for i, rsource in df_source.iterrows():
        source_node = f'source_{i}'
        for j, rsink in df_sink.iterrows():
            sink_node = f'sink_{j}'

            # Get path using path-dependent Dijkstra
            path, adjusted_distance, actual_distance, hop_count = path_dependent_dijkstra(graph, source_node, sink_node)

            # Register path with its details
            path_registry[(i,j)] = {
                'path':path,
                'adjusted_distance':adjusted_distance,
                'actual_distance':actual_distance,
                'hop_count':hop_count,
                'source_id':f'source_id_{i}',
                'sink_id':f'sink_id_{j}'
            }
    return path_registry



    # emission_cost = 100

    # lat_max = max(max(df_sink[sink_lat]),max(df_source[source_lat]))
    # lon_max = max(max(df_sink[sink_lon]),max(df_source[source_lon]))
    # lat_min = min(min(df_sink[sink_lat]),min(df_source[source_lat]))
    # lon_min = min(min(df_sink[sink_lon]),min(df_source[source_lon]))





def path_based_mcf_model(df_source, df_sink, path_registry, emission_cost, source_id, sink_id, capture_cost, transport_method, transport_cost, quantity_transport_cost, stock_cost):


    df_source = df_source.set_index(df_source[source_id])
    df_sink = df_sink.set_index(df_sink[sink_id])

    # Init problem
    prob = pulp.LpProblem("PathBasedMCF", pulp.LpMinimize)

    # Create flow variables for each path 
    path_vars = {}
    for (i,j), path_info in path_registry.items():
        var_name = f'Flow_from_{i}_to_{j}'
        path_vars[(i,j)] = pulp.LpVariable(var_name, 0, None, pulp.LpContinuous)

    # Create flow variables for emissions to atmosphere
    atmo_vars = {}
    for i,row in df_source.iterrows():
        var_name = f"Flow_from_{i}_to_atmo"
        atmo_vars[i] = pulp.LpVariable(var_name, 0, None, pulp.LpContinuous)

    # Cost function: placewise linear segments for path-based transportation costs
    path_segment_vars = {}

    for (i,j), path_info in path_registry.items():
        segments = []
        # The scaled distance cost uses actual_distance froim the path
        distance = path_info['actual_distance']


        # Consider the distance factor 
        if distance < 180:
            distance *= transport_cost[transport_method]['less_180']
        elif distance < 500:
            distance *= transport_cost[transport_method]['range_180_500']
        elif distance < 750:
            distance *= transport_cost[transport_method]['range_500_750']
        elif distance < 1500:
            distance *= transport_cost[transport_method]['range_750_1500']
        else:
            distance *= transport_cost[transport_method]['more_1500']


        # Cost segments using path distnace
        # cost_segments = [
        #     (0,50000, transport_cost_0_50000 * distance),
        #     (50000,100000, transport_cost_50000_100000 * distance),
        #     (100000,250000, transport_cost_100000_250000 * distance),
        #     (250000,500000, transport_cost_250000_500000 * distance),
        #     (500000,1000000, transport_cost_500000_1000000 * distance),
        #     (1000000,2000000, transport_cost_1000000_2000000 * distance),
        #     (2000000,999999999, transport_cost_2000000_999999999 * distance),
        # ]

        cost_segments = [
            (0,50000, quantity_transport_cost[transport_method][(0,50000)] * distance),
            (50000,100000, quantity_transport_cost[transport_method][(50000,100000)] * distance),
            (100000,250000, quantity_transport_cost[transport_method][(100000,250000)] * distance),
            (250000,500000, quantity_transport_cost[transport_method][(250000,500000)] * distance),
            (500000,1000000, quantity_transport_cost[transport_method][(500000,1000000)] * distance),
            (1000000,2000000, quantity_transport_cost[transport_method][(1000000,2000000)] * distance),
            (2000000,999999999, quantity_transport_cost[transport_method][(2000000,999999999)] * distance),
        ]



        for i_seg, (start, end, slope) in enumerate(cost_segments):
            var = pulp.LpVariable(f'Segment_path_{i}_{j}_{i_seg}', 0, end-start, pulp.LpContinuous)
            segments.append((var, start, end, slope))

        path_segment_vars[(i,j)] = segments

    # Same for atmosphere emission costs
    atmo_segment_vars = {}
    for i,row in df_source.iterrows():
        var = pulp.LpVariable(f'Segment_atmo_{i}', 0, 100000000000000000000000000000000000000000000000000000, pulp.LpContinuous)
        atmo_segment_vars[i] = [(var, 0, 100000000000000000000000000000000000000000000000000000, emission_cost)]

    # Objective: Minimize total cost acrtoss all paths and emissions
    prob += pulp.lpSum([var * slope for (i,j) in path_registry for var, start, end, slope in path_segment_vars[(i,j)]] + 
                       [capture_cost * path_vars[(i,j)] for (i,j) in path_registry] +
                       [stock_cost * path_vars[(i,j)] for (i,j) in path_registry] +
                       [var * slope for i in df_source.index for var, start, end, slope in atmo_segment_vars[i]]), "TotalCost"
    
    # Constraint: all source emission must be extracted from source
    print(atmo_vars)
    for i, rsource in df_source.iterrows():
        prob += (pulp.lpSum([path_vars[(i,j)] for j in df_sink.index if (i,j) in path_vars]) + 
                 atmo_vars[i] == rsource['emission']), f'source_id_{i}_flow_conservation'
        
    # Constraint: link flow variables to their piecewise segments 
    for (i,j) in path_registry:
        prob += path_vars[(i,j)] == pulp.lpSum([var for var, start, end, slope in path_segment_vars[(i,j)]]), f"Piecewise_path_{i}_{j}"
    
    for i, row in df_source.iterrows():
        prob += atmo_vars[i] == pulp.lpSum([var for var, start, end, slope, in atmo_segment_vars[i]]), f'Piecewise_atmo_{i}'
    


    # Optional: capacity constraints for paths
    for (i,j) in path_registry:
        prob += path_vars[(i,j)] <= 10000000000 #000000

    return prob, path_vars, atmo_vars



# def visualize_flow_map(map, network, df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, path_registry, path_vars):
#     breakpoint()
#     path_vars['index'] = path_vars['index'].map(ast.literal_eval)
#     path_vars = {row['index'] : row['flow_name'] for i, row in path_vars.iterrows()}

#     path_registry['idx'] = pd.Series()

#     for i, row in path_registry.iterrows():
#         path_registry.at[i,'idx'] = (row['Unnamed: 0'], row['Unnamed: 1'])
#     path_registry = path_registry.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)
#     path_registry['path'] = path_registry['path'].map(ast.literal_eval)

#     path_registry = path_registry.set_index('idx')
#     path_registry = path_registry.T.to_dict()


#     # Dictionary to track cumulative flow on each edge
#     edge_flows = {}

#     # First calculate cumulative flow on each edge

#     for (i,j), path_info in path_registry.items():
#         flow_amount = path_vars[(i,j)].varValue

#         if flow_amount > 0:
#             path = path_info['path']

#             # Process each edge in the path
#             for k in range(len(path) - 1):
#                 edge = (path[k], path[k+1])

#                 # Add flow to this edge
#                 if edge in edge_flows:
#                     edge_flows[edge] += flow_amount
#                 else:
#                     edge_flows[edge] = flow_amount

#     # Visualize each edge with its cumulative flow
#     for edge, flow in edge_flows.items():
#         start_node, end_node = edge

#         # Get coordinates for start node
#         if "source" in start_node:
#             idx = int(start_node.split('_')[1])
#             start_lat = df_source.loc[idx][source_lat] ####
#             start_lon = df_source.loc[idx][source_lon] ####
#         else:
#             idx = int(end_node.split('_')[1])
#             start_lat = df_sink.loc[idx][sink_lat]
#             start_lon = df_sink.loc[idx][sink_lon]
    
#         # Get coordinates for end node
#         if "source" in end_node:
#             idx = int(end_node.split('_')[1])
#             end_lat = df_source.loc[idx][source_lat]
#             end_lon = df_source.loc[idx][source_lon]
#         else:
#             idx = int(end_node.split('_')[1])
#             end_lat = df_sink.loc[idx][sink_lat]
#             end_lon = df_sink.loc[idx][sink_lon]


#         # Calculate edge weight with cumulative flow tooltip
#         weight = 2 + (flow / 1000000) # Adjust as needed

#         folium.PolyLine(
#             locations=[(start_lat, start_lon), (end_lat, end_lon)],
#             weight=weight,
#             color='green',
#             opacity=0.7,
#             tooltip=f"Flow: {flow:.2f} units \n{start_node}  ==>  {end_node}"
#         ).add_to(map)

#     # Add detail information as a control panel
#     html_content = """
#     <div style="position: fixed;
#                 bottom: 50px; right: 50px; width: 280px;
#                 border:2px solid grey; z-index:9999; font-size:12px;
#                 background-color:white; padding:10px;
#                 opacity:0.8;">
#     <h4>Edge Details</h4>
#     <table>
#         <tr><th>Edge</th><th>Flow</th></tr>
#     """

#     # Sort edges by flow amount for better readability
#     sorted_edges = sorted(edge_flows.items(), key=lambda x: x[1], reverse=True)

#     for edge, flow in sorted_edges:
#         start, end = edge
#         html_content += f"<tr><td>{start} ==> {end}</td><td>{flow:.2f}</td></tr>"

#     html_content += """
#     </table>
#     </div>
#     """
        
#     # Add the HTML content to the map
#     folium.Element(html_content).add_to(map)

#     return map



def visualize_flow_map(map_obj, flow_df, df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon):
    """
    Visualize flow map using DataFrame input format
    
    Args:
        map_obj: Folium map object
        df_source: DataFrame with source node information (must contain 'lat' and 'lon' columns)
        df_sink: DataFrame with sink node information (must contain 'latitude' and 'longitude' columns)
        flow_df: DataFrame containing flow information with columns:
                - source_id: source node ID
                - sink_id: sink node ID
                - co2_transported: flow amount
                - dijkstra_path: list representing the path from source to sink
    """
    flow_df['dijkstra_path'] = flow_df['dijkstra_path'].map(ast.literal_eval)

    # Dictionary to track cumulative flow on each edge
    edge_flows = {}

    # First calculate cumulative flow on each edge
    for _, row in flow_df.iterrows():
        flow_amount = row['co2_transported']
        path = row['dijkstra_path']

        if flow_amount > 0:
            # Process each edge in the path
            for k in range(len(path) - 1):
                edge = (path[k], path[k+1])

                # Add flow to this edge
                if edge in edge_flows:
                    edge_flows[edge] += flow_amount
                else:
                    edge_flows[edge] = flow_amount
    
    # Visualize each edge with its cumulative flow
    for edge, flow in edge_flows.items():
        start_node, end_node = edge

        # Get coordinates for start node

        if "source" in start_node:
            idx = int(start_node.split('_')[1])
            start_lat = df_source.loc[df_source[source_id] == idx, source_lat].values[0]
            start_lon = df_source.loc[df_source[source_id] == idx, source_lon].values[0]
        else:
            idx = int(start_node.split('_')[1])
            start_lat = df_sink.loc[df_sink[sink_id] == idx, sink_lat].values[0]
            start_lon = df_sink.loc[df_sink[sink_id] == idx, sink_lon].values[0]
        
        # Get coordinates for end node
        if end_node != "Atmosphere":
            if "source" in end_node:
                idx = int(end_node.split('_')[1])
                end_lat = df_source.loc[df_source[source_id] == idx, source_lat].values[0]
                end_lon = df_source.loc[df_source[source_id] == idx, source_lon].values[0]
            else:
                idx = int(end_node.split('_')[1])
                end_lat = df_sink.loc[df_sink[sink_id] == idx, sink_lat].values[0]
                end_lon = df_sink.loc[df_sink[sink_id] == idx, sink_lon].values[0]
        else:
            idx = int(start_node.split('_')[1])
            end_lat = df_source.loc[df_source[source_id] == idx, source_lat].values[0]
            end_lon = df_source.loc[df_source[source_id] == idx, source_lon].values[0]

            
        
        # Calculate edge weight with cumulative flow tooltip
        weight = 2 + (flow / 1000000) # Adjust as needed
        folium.PolyLine(
            locations=[(start_lat, start_lon), (end_lat, end_lon)],
            weight=weight,
            color='blue',
            opacity=0.7,
            tooltip=f"Flow: {flow:.2f} units \n{start_node} ==> {end_node}"
        ).add_to(map_obj)
    
    # Add detail information as a control panel
    html_content = """
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 280px;
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; padding:10px;
                opacity:0.8;">
    <h4>Edge Details</h4>
    <table>
        <tr><th>Edge</th><th>Flow</th></tr>
    """

    # Sort edges by flow amount for better readability
    sorted_edges = sorted(edge_flows.items(), key=lambda x: x[1], reverse=True)

    for edge, flow in sorted_edges:
        start, end = edge
        html_content += f"<tr><td>{start} ==> {end}</td><td>{flow:.2f}</td></tr>"

    html_content += """
    </table>
    </div>
    """
        
    # Add the HTML content to the map
    folium.Element(html_content).add_to(map_obj)

    return map_obj