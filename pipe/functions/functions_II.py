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