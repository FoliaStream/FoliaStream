import pandas as pd
import numpy as np 
import requests
import polyline
import pulp

from requests.models import PreparedRequest
from geopy.distance import geodesic


# //////////////////////////////////////////////////////
#                  FUNCTIONS II LEVEL
# //////////////////////////////////////////////////////

# Structure source data for export
def export_data_structure(df, index, key_col):

    actions = [
        {
            '_op_type':'index',
            '_index': index,
            str(f'_{key_col}'): row[key_col],
            str(f'_{index}'): {
                col : row[col] for col in df.columns
            }
        } for _, row in df.iterrows()
    ]

    return actions



# Generate url for API query 
def request_url(url, params):
    
    request = PreparedRequest()
    request.prepare_url(url, params)

    return request.url
