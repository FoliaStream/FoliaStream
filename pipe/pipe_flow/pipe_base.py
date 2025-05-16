
import logging
import os
import yaml 

from dataclasses import dataclass
from abc import ABC, abstractmethod


#///////////////////
#   CONFIGURATION 
#///////////////////

@dataclass
class Config:


# Variables

    # Filters
    country: str
    year: str
    sector: str
    gas: str
    cluster_bool: bool

    # Values
    capture_cost: float
    emission_cost: float

    # Paths
    out_csv_path: str
    out_fig_path: str
    in_sink_path: str

    # Elastic
    es_host: str
    es_auth: tuple

    # API
    source_api_url: str 
    limit: int

    # Index
    source_index: str
    sink_index: str
    matrix_index: str
    network_results_index: str
    clusters_centroids_index: str

    # Files
    output_map: str
    output_network_map: str
    output_network: str
    output_results_calc: str

    # Columns
    source_id_col: str
    source_emit_col: str
    source_lat_col: str
    source_lon_col: str
    source_cluster_col: str

    sink_id_col: str
    sink_capacity_col: str
    sink_lat_col: str
    sink_lon_col: str
    sink_country_col: str

    centroid_id_col: str
    centroid_lat_col: str
    centroid_lon_col: str

    # Mappings
    mappings_source: dict
    mappings_source_clust: dict
    mappings_sink: dict
    mappings_centroids: dict
    
    # Query 
    query: dict 
    query_matrix: dict
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    




#///////////////////
#   PARSE CONFIG 
#///////////////////

def parse_config():
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(pipeline_dir), "config")

    base_config = os.path.join(config_path,"base.yaml")
    with open(base_config, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    input_config = os.path.join(config_path, "case.yaml")
    if os.path.exists(input_config):
        with open(input_config, "r") as file:
            data_input = yaml.load(file, Loader=yaml.FullLoader)
        data.update(data_input)
        
    config = Config.from_dict(data)
    return config







#///////////////////
#     PIPE BASE
#///////////////////

class PipelineBase(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = parse_config()
    
    @abstractmethod
    def run(self):
        pass
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = parse_config()

    @abstractmethod
    def run(self):
        pass