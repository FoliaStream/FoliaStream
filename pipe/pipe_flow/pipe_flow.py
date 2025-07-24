import pandas as pd

from pipe.pipe_flow.pipe_base import PipelineBase
from pipe.functions.functions_I import clean_folder, create_folder, source_import_api, source_edit, csv_import, sink_edit, nodes_map, create_matrix_cc, network_optimization, network_map, network_optimization_klust, network_map_klust, network_optimization_levelized, network_optimization_klust_levelized, network_optimization_dijkstra, network_map_dijkstra, create_matrix_dac


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

        # Step . Clean folders
        self.call_clean_folder(
            case_paths=[
            str(str(s.out_csv_path_temp)),
            str(str(s.out_fig_path_temp)),
            str(str(s.out_csv_path_final)),
            str(str(s.out_fig_path_final))],
            )


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
            out_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.matrix_out}"),
            out_path_dac=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.dac_out}")
        )

        # Step . Network optimization
        self.call_network_optimization(
            source_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}"),
            sink_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}"),
            matrix_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.matrix_out}"), 
            out_path=str(f"{s.out_csv_path_final}{s.country}__{s.year}__{s.sector}/{s.network_results}"),
            out_path_registry=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.path_registry}"),
            out_path_vars=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.path_vars}"),
            dac_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.dac_out}"),
            totals_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.totals_out}")
            )

        # Step . Network map
        self.call_network_map(
            source_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.source_raw}"),
            sink_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.sink_raw}"),
            network_path=str(f"{s.out_csv_path_final}{s.country}__{s.year}__{s.sector}/{s.network_results}"), 
            out_path=str(f"{s.out_fig_path_final}{s.country}__{s.year}__{s.sector}/{s.network_map_out}"),
            in_path_registry=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.path_registry}"),
            in_path_vars=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.path_vars}"),
            dac_path=str(f"{s.out_csv_path_temp}{s.country}__{s.year}__{s.sector}/{s.dac_out}")
        )





#/////////////////////////////////////
#           CALL FUNCTIONS
#/////////////////////////////////////

    # Step . Clean folder
    def call_clean_folder(self, case_paths: list) -> any:

        s = self.config

        # Compile
        for path in case_paths:
            folder = clean_folder(str(path))

            print(f"\n{path}")

        # Success
        print(f"\n------------------- Clean folders -------------------\n")
        return case_paths, folder



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
                                 s.source_lon_col,
                                 s.source_name_col)

        # Export
        source_out.to_csv(out_path, index=False)

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
                             s.country,
                             s.sink_name_col)
        
        # Export
        sink_out.to_csv(out_path, index=False)

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
    def call_create_matrix(self, source_path, sink_path, out_path, out_path_dac):

        s = self.config

        # Import 
        sink_in = csv_import(sink_path)
        source_in = csv_import(source_path)

        # Compile

        if s.capture_method == 'Carbon Capture (CC)':
            matrix = create_matrix_cc(source_in,
                                sink_in,
                                s.source_id_col,
                                s.source_lat_col,
                                s.source_lon_col,
                                s.sink_id_col,
                                s.sink_lat_col,
                                s.sink_lon_col, 
                                s.emission_cost,
                                s.capture_cost,
                                s.osrm_api_table_url,
                                s.transport_cost,
                                s.transport_method, 
                                s.batch_size)
        
            # Export
            matrix.to_csv(out_path)


        elif s.capture_method == 'Direct Air Capture (DAC)':
            matrix, source_dac = create_matrix_dac(source_in,
                                sink_in,
                                s.source_id_col,
                                s.source_lat_col,
                                s.source_lon_col,
                                s.sink_id_col,
                                s.sink_lat_col,
                                s.sink_lon_col, 
                                s.emission_cost,
                                s.capture_cost,
                                s.osrm_api_table_url,
                                s.transport_cost,
                                s.transport_method, 
                                s.batch_size,
                                s.source_emit_col,
                                s.source_name_col)
            
            # Export
            matrix.to_csv(out_path)
            source_dac.to_csv(out_path_dac)

        

        # Success
        print(f"\n------------------- Cost matrix created -------------------\n")
        return matrix

    

    # Step . Network optimization
    def call_network_optimization(self, source_path, sink_path, matrix_path, out_path, out_path_registry, out_path_vars, dac_path, totals_path):

        s = self.config

        # Import 
        sink_in = csv_import(sink_path)
        source_in = csv_import(source_path)
        matrix_in = csv_import(matrix_path)

        if s.capture_method == 'Carbon Capture (CC)':
            source_in = csv_import(source_path)
        elif s.capture_method == 'Direct Air Capture (DAC)':
            source_in = csv_import(dac_path)
        else:
            pass


        # Compile

        if s.network_type == 'Direct connection':
            network_results, totals = network_optimization_levelized(source_in,
                                                sink_in,
                                                matrix_in,
                                                s.source_id_col,
                                                s.sink_id_col,
                                                s.source_emit_col,
                                                s.sink_capacity_col, 
                                                s.emission_cost,
                                                s.transport_method,
                                                s.quantity_cost_segments,
                                                s.capture_cost,
                                                s.stock_cost)


        elif s.network_type == '1k-cluster connection':

            network_results, totals = network_optimization_klust_levelized(source_in,
                                                sink_in,
                                                matrix_in,
                                                s.source_id_col,
                                                s.sink_id_col,
                                                s.source_emit_col,
                                                s.sink_capacity_col,
                                                s.osrm_api_table_url, 
                                                s.transport_method, 
                                                s.transport_cost, 
                                                s.emission_cost, 
                                                s.capture_cost, 
                                                s.quantity_cost_segments,
                                                s.stock_cost)

        
        elif s.network_type == 'Dijkstra connection':
            network_results, path_registry, path_vars, totals = network_optimization_dijkstra(source_in,
                                                            sink_in,
                                                            s.source_id_col,
                                                            s.sink_id_col,
                                                            s.source_lat_col,
                                                            s.sink_lat_col,
                                                            s.source_lon_col,
                                                            s.sink_lon_col,
                                                            s.emission_cost,
                                                            s.capture_cost,
                                                            s.transport_method,
                                                            s.transport_cost, 
                                                            s.quantity_cost_segments, 
                                                            s.stock_cost)


        # Export
        network_results.to_csv(out_path, index=False)
        totals.to_csv(totals_path, index=False)

        if s.network_type == 'Dijkstra connection':

            path_registry = path_registry.T
            path_registry.to_csv(out_path_registry)
            path_vars.to_csv(out_path_vars, index = False)
        
        # Success
        print(f"\n------------------- Network optimized -------------------\n")
        return network_results
    

    # Step . Network map
    def call_network_map(self, source_path, sink_path, network_path, out_path, in_path_registry, in_path_vars, dac_path):

        s = self.config

        # Import
        sink_in = csv_import(sink_path)
        network_results = csv_import(network_path)

        if s.capture_method == 'Carbon Capture (CC)':
            source_in = csv_import(source_path)
        elif s.capture_method == 'Direct Air Capture (DAC)':
            source_in = csv_import(dac_path)
        else:
            pass
        
        if s.network_type == 'Dijkstra connection':
            path_registry = pd.read_csv(in_path_registry)
            path_vars = pd.read_csv(in_path_vars)

        # Compile
        if s.network_type=='Direct connection':
            map = network_map(network_results, 
                            source_in,
                            sink_in,
                            s.source_id_col,
                            s.sink_id_col,
                            s.source_lat_col, 
                            s.sink_lat_col,
                            s.source_lon_col,
                            s.sink_lon_col)
        elif s.network_type == '1k-cluster connection':
            map = network_map_klust(network_results, 
                            source_in,
                            sink_in,
                            s.source_id_col,
                            s.sink_id_col,
                            s.source_lat_col, 
                            s.sink_lat_col,
                            s.source_lon_col,
                            s.sink_lon_col)
        elif s.network_type == 'Dijkstra connection':
            map = network_map_dijkstra(network_results, 
                            source_in,
                            sink_in,
                            s.source_id_col,
                            s.sink_id_col,
                            s.source_lat_col, 
                            s.sink_lat_col,
                            s.source_lon_col,
                            s.sink_lon_col,
                            path_registry,
                            path_vars)

        # Export
        map.save(out_path)

        # Success
        print(f"\n------------------- Network map created -------------------\n")
        return map