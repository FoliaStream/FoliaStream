import pandas as pd

from pipe.pipe_flow.pipe_base import PipelineBase
from pipe.functions.functions_I import create_folder, source_import_api, source_edit, csv_import, sink_edit, nodes_map, create_matrix, network_optimization, network_map


import warnings
warnings.filterwarnings('ignore')


class PipelineFlow(PipelineBase):

    # Pipeline initialization
    def __init__(self) -> None:
        super().__init__()

    
    # RUN
    def run(self):

        s = self.config

        # START

        # Step . Create folders
        self.call_create_folder(
            case_paths=[
            str(str(s.out_csv_path_temp) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/"),
            str(str(s.out_fig_path_temp) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/"),
            str(str(s.out_csv_path_final) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/"),
            str(str(s.out_fig_path_final) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/")],
            )
        
        # Step . Source load
        self.call_source_load(
            api_url = s.source_api_url,
            api_params = {'limit' : s.limit,
                          'gas' : s.gas,
                          'countries' : s.country,
                          'year' : s.year,
                          'subsectors' : s.sector},
            out_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}")
        )

        # Step . Sink load
        self.call_sink_load(
            in_path=str(f"{s.in_sink_path}"),
            out_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}")
        )

        # Step . Nodes map
        self.call_nodes_map(
            sink_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}"),
            source_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}"),
            out_path=str(f"{s.out_fig_path_final}{s.country}__{s.year}__{s.sector}/{s.nodes_map_out}")
        )

        # Step . Matrix
        self.call_create_matrix(
            source_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}"),
            sink_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}"),
            out_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.matrix_out}")
        )

        # Step . Network optimization
        self.call_network_optimization(
            source_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}"),
            sink_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}"),
            matrix_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.matrix_out}"), 
            out_path=str(f"{s.out_csv_path_final}{s.country}__{s.year}__{s.sector}/{s.network_results}"))

        # Step . Network map
        self.call_network_map(
            source_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}"),
            sink_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}"),
            network_path=str(f"{s.out_csv_path_final}{s.country}__{s.year}__{s.sector}/{s.network_results}"), 
            out_path=str(f"{s.out_fig_path_final}{s.country}__{s.year}__{s.sector}/{s.network_map_out}")
        )





#/////////////////////////////////////
#           CALL FUNCTIONS
#/////////////////////////////////////

    # Step . Create folder
    def call_create_folder(self, case_paths: list) -> any:

        s = self.config

        # Compile
        for path in case_paths:
            folder = create_folder(str(path))

            print(f"\n{path}")

        # Success
        print(f"\n------------------- Created folders -------------------\n")
        return case_paths, folder
    

    # Step . Source load
    def call_source_load(self, api_url, api_params, out_path):

        s = self.config

        # Import
        source_in = source_import_api(api_url, 
                                      api_params)

        # Compile
        source_out = source_edit(source_in, 
                                 s.source_id_col, 
                                 s.source_emit_col, 
                                 s.source_lat_col, 
                                 s.source_lon_col)

        # Export
        source_out.to_csv(out_path)

        # Success
        print(f"\n------------------- Source data loaded -------------------\n")
        return source_out
    

    # Step . Sink load
    def call_sink_load(self, in_path, out_path):

        s = self.config

        # Import 
        sink_in = csv_import(in_path) # not necessary but ok for structure

        # Compile
        sink_out = sink_edit(sink_in, 
                             s.sink_id_col, 
                             s.sink_country_col, 
                             s.sink_capacity_col, 
                             s.sink_lat_col, 
                             s.sink_lon_col, 
                             s.country)
        
        # Export
        sink_out.to_csv(out_path)

        # Success
        print(f"\n------------------- Sink data loaded -------------------\n")
        return sink_out
    

    # Step . Nodes map
    def call_nodes_map(self, source_path, sink_path, out_path):
        
        s = self.config

        # Import 
        sink_in = csv_import(sink_path)
        source_in = csv_import(source_path)

        # Compile
        map = nodes_map(source_in,
                        sink_in,
                        s.source_id_col,
                        s.source_lat_col,
                        s.source_lon_col,
                        s.sink_id_col,
                        s.sink_lat_col,
                        s.sink_lon_col)

        # Export
        map.save(out_path)

        # Success
        print(f"\n------------------- Nodes map created -------------------\n")
        return map


    # Step . Cost matrix
    def call_create_matrix(self, source_path, sink_path, out_path):

        s = self.config

        # Import 
        sink_in = csv_import(sink_path)
        source_in = csv_import(source_path)

        # Compile
        matrix = create_matrix(source_in,
                               sink_in,
                               s.source_id_col,
                               s.source_lat_col,
                               s.source_lon_col,
                               s.sink_id_col,
                               s.sink_lat_col,
                               s.sink_lon_col, 
                               s.emission_cost,
                               s.capture_cost)

        # Export
        matrix.to_csv(out_path)

        # Success
        print(f"\n------------------- Cost matrix created -------------------\n")
        return matrix
    

    # Step . Network optimization
    def call_network_optimization(self, source_path, sink_path, matrix_path, out_path):

        s = self.config

        # Import 
        sink_in = csv_import(sink_path)
        source_in = csv_import(source_path)
        matrix_in = csv_import(matrix_path)

        # Compile
        network_results = network_optimization(source_in,
                                               sink_in,
                                               matrix_in,
                                               s.source_id_col,
                                               s.sink_id_col,
                                               s.source_emit_col,
                                               s.sink_capacity_col)

        # Export
        network_results.to_csv(out_path)
        
        # Success
        print(f"\n------------------- Network optimized -------------------\n")
        return network_results
    

    # Step . Network map
    def call_network_map(self, source_path, sink_path, network_path, out_path):

        s = self.config

        # Import
        source_in = csv_import(source_path)
        sink_in = csv_import(sink_path)
        network_results = csv_import(network_path)

        # Compile
        map = network_map(network_results, 
                          source_in,
                          sink_in,
                          s.source_id_col,
                          s.sink_id_col,
                          s.source_lat_col, 
                          s.sink_lat_col,
                          s.source_lon_col,
                          s.sink_lon_col)

        # Export
        map.save(out_path)

        # Success
        print(f"\n------------------- Network map created -------------------\n")
        return map