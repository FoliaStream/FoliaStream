
from requests.models import PreparedRequest
from sklearn.cluster import KMeans
from kneed import KneeLocator 

import pandas as pd
import numpy as np


# ////////////////////////////////////////////////
#                   FUNCTIONS II
# ////////////////////////////////////////////////


# Generate url for API query 
def request_url(url, params):
    
    request = PreparedRequest()
    request.prepare_url(url, params)

    return request.url


# //////////////////////////
# ABSOLUTELY TO BE REWRITTEN
# create single function to export all info, loop over columns rather than creating a function for each different data structure exported

# Structure source data for export
def export_data_structure(df, index, id_col, capacity_col, lat_col, lon_col):

    actions = []
    for i in range(len(df)):
        if df.iloc[i][id_col] is not None:

            actions.append(
                {
                    '_op_type': 'index',  
                    '_index': index,
                    str(f'_{id_col}'): df.iloc[i][id_col],  
                    str(f'_{index}'): {
                        id_col: df.iloc[i][id_col],
                        lat_col: df.iloc[i][lat_col],
                        lon_col: df.iloc[i][lon_col],
                        capacity_col: df.iloc[i][capacity_col]
                    }
                }
            )
            
    return actions


# Structure source data for export with cluster info
def export_data_structure_cluster(df, index, id_col, capacity_col, lat_col, lon_col, cluster_col):

    actions = []
    for i in range(len(df)):
        if df.iloc[i][id_col] is not None:

            actions.append(
                {
                    '_op_type': 'index',  
                    '_index': index,
                    str(f'_{id_col}'): df.iloc[i][id_col],  
                    str(f'_{index}'): {
                        id_col: df.iloc[i][id_col],
                        lat_col: df.iloc[i][lat_col],
                        lon_col: df.iloc[i][lon_col],
                        capacity_col: df.iloc[i][capacity_col], 
                        cluster_col: df.iloc[i][cluster_col]
                    }
                }
            )
    return actions


# Structure centroids data
def export_data_structure_centroids(df, index, id_col, lat_col, lon_col):

    actions = []
    for i in range(len(df)):
        if df.iloc[i][id_col] is not None:

            actions.append(
                {
                    '_op_type': 'index',
                    '_index':index,
                    str(f'_{id_col}'):df.iloc[i][id_col],
                    str(f'_{index}'):{
                        id_col: df.iloc[i][id_col],
                        lat_col: df.iloc[i][lat_col],
                        lon_col: df.iloc[i][lon_col]
                    }
                }
            )

    return actions
    



# //////////////////////////


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

    return elbow_point


def create_clusters(data, x_ax, y_ax, k, weight):

    coord_x = np.zeros(len(data))
    coord_y = np.zeros(len(data))

    for i, row in data.iterrows():
        coord_x[i] = data[x_ax][i]
        coord_y[i] = data[y_ax][i]
        
    X = np.array(list(zip(coord_x,coord_y))).reshape(len(coord_x),2)

    kmeans_model = KMeans(n_clusters=k).fit(X, sample_weight=data[weight]) 

    centroids = kmeans_model.cluster_centers_.tolist()
    centroids = pd.DataFrame(centroids, columns=[x_ax, y_ax])
    centroids['id'] = pd.Series(centroids.index)
    labels = kmeans_model.labels_.tolist()

    return centroids, labels
