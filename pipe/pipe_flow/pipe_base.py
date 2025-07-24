import logging
import os
import yaml

from dataclasses import dataclass
from abc import ABC, abstractmethod

#//////////////////////////////////
#          CONFIGURATION           
#//////////////////////////////////

@dataclass
class Config:

# Variables

    # Filters
    country: str
    year: str
    sector: str

    # Transport
    transport_method: str
    transport_cost: dict
    quantity_cost_segments: dict

    # Capture
    capture_method: str

    # Values
    gas: str
    limit: int
    capture_cost: int
    emission_cost: int
    network_type: str
    stock_cost: int

    # Paths
    out_csv_path_temp: str
    out_fig_path_temp: str
    out_csv_path_final: str
    out_fig_path_final: str
    in_sink_path: str
    

    # API
    source_api_url: str
    osrm_api_url: str
    osrm_api_table_url: str
    batch_size: int

    # File names
    optimal_network: str
    source_raw: str
    sink_raw: str
    nodes_map_out: str
    matrix_out: str
    network_results: str
    network_map_out: str
    path_registry: str
    path_vars: str
    dac_out: str
    totals_out : str

    # Columns
    sink_id_col: str
    sink_capacity_col: str
    sink_lat_col: str
    sink_lon_col: str
    sink_country_col: str
    sink_name_col: str

    source_id_col: str
    source_emit_col: str
    source_lat_col: str
    source_lon_col: str
    source_name_col: str


    @classmethod
    def from_dict(cls, data):
        return cls(**data)


#//////////////////////////////////
#          PARSE CONFIG           
#//////////////////////////////////

def parse_config():
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(pipeline_dir), "config")

    base_config = os.path.join(config_path, "base.yaml")
    with open(base_config, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    input_config = os.path.join(config_path, "case.yaml")
    if os.path.exists(input_config):
        with open(input_config, "r") as file:
            data_input = yaml.load(file, Loader=yaml.FullLoader)
        data.update(data_input)
    
    config = Config.from_dict(data)
    return config



#//////////////////////////////////
#            PIPE BASE           
#//////////////////////////////////

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