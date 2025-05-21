from requests.models import PreparedRequest
from sklearn.cluster import KMeans
from kneed import KneeLocator 
from geopy.distance import geodesic

import pandas as pd
import numpy as np

import requests

# ////////////////////////////////////////////////
#                   FUNCTIONS II
# ////////////////////////////////////////////////

# Generate url for API query 
def request_url(url, params):
    
    request = PreparedRequest()
    request.prepare_url(url, params)

    return request.url


# Distance matrix


def distance_matrix(url, source, sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, transport_method):

    matrix = pd.DataFrame()

    if transport_method == 'pipe':

        for i, rsink in sink.iterrows():
            for j, rsource in source.iterrows():
                matrix.at[i,j] = geodesic((float(rsource[source_lat]),float(rsource[source_lon])), (float(rsink[sink_lat]),float(rsink[sink_lon]))).km


    elif transport_method == 'truck_ship':

        sink = sink.rename(columns={'latitude':'lat', 'longitude':'lon'})
        combined_df = pd.concat([source, sink], ignore_index=True)
        all_coords  = ";".join([f"{lon},{lat}" for lat, lon in zip(combined_df['lat'], combined_df['lon'])])

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




# Cost matrix (careful, the transport cost is not dependent on the quantity transported --> multiply this cost by the total number of tons of relative source // adjust optimization model to consider transport cost dependent on the number of tons transported, more correct)
def cost_matrix(matrix, method, transport_cost, emission_cost, capture_cost):

    for col in matrix.columns:

        for id in matrix[col].index:

            if matrix.at[id,col]<180:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['less_180']) + capture_cost

            elif matrix.at[id,col] >= 180 and matrix.at[id,col] < 500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_180_500']) + capture_cost
            
            elif matrix.at[id,col] >= 500 and matrix.at[id,col] < 750:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_500_750']) + capture_cost

            elif matrix.at[id,col] >= 750 and matrix.at[id,col] < 1500:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['range_750_1500']) + capture_cost

            else:
                matrix.at[id,col] = matrix.at[id,col]*float(transport_cost[method]['more_1500']) + capture_cost
            
    atmosphere_row = pd.DataFrame({col:emission_cost for col in matrix.columns}, index=["Atmosphere"])

    matrix = pd.concat([matrix, atmosphere_row], axis=0)

    return matrix
