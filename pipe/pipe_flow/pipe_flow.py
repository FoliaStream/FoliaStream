
import pandas as pd
import pathlib as Pth

from .pipe_base import PipelineBase
from functions_I import create_folder, source_import_api, source_edit, source_export, sink_edit, sink_export, import_data, nodes_map, create_matrix, matrix_export, matrix_import, network_opt_min, network_export, network_import, network_map, results_calc, create_cluster, centroids_export, network_opt_min_inter
from elasticsearch import Elasticsearch, helpers

import warnings
warnings.filterwarnings('ignore')


#/////////////////////////////////////////////////////
#                   PIPELINE FLOW
#/////////////////////////////////////////////////////


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
            str(str(s.out_csv_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/"),
            str(str(s.out_fig_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/")]
            )

        # Step . Source load
        self.call_source_load(
            api_url = s.source_api_url,
            api_params = {'limit' : s.limit,
                          'gas' : s.gas,
                          'countries' : s.country,
                          'year' : s.year,
                          'subsectors' : s.sector},
            host = s.es_host,
            auth = s.es_auth,
            mappings = s.mappings_source,
            source_index = s.source_index
            )
        
        # Step . Clustering
        self.call_create_cluster(
            host = s.es_host, 
            auth = s.es_auth, 
            source_index = s.source_index, 
            query = s.query, 
            cluster_bool = s.cluster_bool, 
            mappings_source_clust = s.mappings_source_clust,
            centroids_index = s.clusters_centroids_index,
            mappings_centroids=s.mappings_centroids)
        
        # Step . Sink load
        self.call_sink_load(
            host = s.es_host,
            auth = s.es_auth,
            mappings = s.mappings_sink,
            sink_index = s.sink_index
            )

        # Step . Nodes map
        self.call_nodes_map(
            host = s.es_host,
            auth = s.es_auth,
            source_index = s.source_index,
            sink_index = s.sink_index, 
            query = s.query,
            output_path = str(str(s.out_fig_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/" + s.output_map),
            cluster_bool=s.cluster_bool,
            centroids_index=s.clusters_centroids_index
            )

        # Step . Matrix
        self.call_create_matrix(
            host = s.es_host,
            auth = s.es_auth,
            source_index = s.source_index,
            sink_index = s.sink_index, 
            matrix_index = s.matrix_index,
            query = s.query,
            capture_cost = s.capture_cost,
            emission_cost = s.emission_cost,
            cluster_bool=s.cluster_bool,
            centroids_index=s.clusters_centroids_index
            )

        # Step . Model
        self.call_network_opt_min(
            host=s.es_host,
            auth=s.es_auth,
            source_index=s.source_index,
            sink_index=s.sink_index,
            matrix_index=s.matrix_index,
            query=s.query_matrix,
            output_path=str(str(s.out_csv_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/" + s.output_network),
            cluster_bool = s.cluster_bool, 
            centroids_index = s.clusters_centroids_index
        )

        # Step . Network map
        self.call_network_map(
            host=s.es_host,
            auth=s.es_auth,
            query=s.query,
            network_index=s.network_results_index,
            source_index=s.source_index,
            sink_index=s.sink_index,
            source_id=s.source_id_col,
            sink_id=s.sink_id_col,
            source_lat=s.source_lat_col,
            sink_lat=s.sink_lat_col,
            source_lon=s.source_lon_col,
            sink_lon=s.sink_lon_col,
            output_path=str(str(s.out_fig_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/" + s.output_network_map)
        )

        # Step . Results calculations
        self.call_results_calc(host=s.es_host,
                               auth=s.es_auth, 
                               index=s.network_results_index,
                               source_id=s.source_id_col,
                               sink_id=s.sink_id_col, 
                               output_path=str(str(s.out_csv_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/" + s.output_results_calc)
                               )



#/////////////////////////////////////////////////////
#                   CALL FUNCTIONS
#/////////////////////////////////////////////////////

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
    def call_source_load(self, api_url, api_params, host, auth, mappings, source_index):

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
        es_success, es_fail = source_export(source_out, 
                                            host, 
                                            auth, 
                                            source_index, 
                                            mappings, 
                                            s.source_id_col, 
                                            s.source_emit_col, 
                                            s.source_lat_col, 
                                            s.source_lon_col)        

        # Success
        print(f" Successfully indexed {es_success} \n Failed indexed {es_fail}")
        print(f"\n------------------- Source loaded -------------------\n")
        return source_out, es_success
        

    # Step . Create cluster
    def call_create_cluster(self, host, auth, source_index, query, cluster_bool, mappings_source_clust, centroids_index, mappings_centroids):

        s = self.config

        # Import source
        df_source = import_data(host, 
                                auth, 
                                source_index, 
                                query, 
                                [s.source_id_col, 
                                s.source_emit_col, 
                                s.source_lat_col, 
                                s.source_lon_col])
        
        if cluster_bool == True:

            # Compile
            centroids, clustered_df = create_cluster(df_source, 
                                                    s.source_lat_col, 
                                                    s.source_lon_col, 
                                                    s.source_emit_col, 
                                                    s.source_cluster_col)

            # Export
            es_success, es_fail = source_export(clustered_df, 
                                                host, 
                                                auth,
                                                source_index,
                                                mappings_source_clust,
                                                s.source_id_col,
                                                s.source_emit_col,
                                                s.source_lat_col,
                                                s.source_lon_col,
                                                s.source_cluster_col)
            
            es_success_clust, es_fail_clust = centroids_export(centroids,
                                                               host,
                                                               auth,
                                                               centroids_index,
                                                               mappings_centroids,
                                                               s.centroid_id_col,
                                                               s.centroid_lat_col,
                                                               s.centroid_lon_col)
            
            # Success
            print(centroids)
            print(f"\n------------------- Clusters loaded -------------------\n")
            return centroids

        else:
            pass



    # Step . Sink load
    def call_sink_load(self, host, auth, mappings, sink_index):

        s = self.config

        # Import 
        sink_in = pd.read_csv(s.in_sink_path)

        # Compile
        sink_out = sink_edit(sink_in, 
                             s.sink_id_col, 
                             s.sink_country_col, 
                             s.sink_capacity_col, 
                             s.sink_lat_col, 
                             s.sink_lon_col, 
                             s.country)

        # Export
        es_success, es_fail = sink_export(sink_out, 
                                          host, 
                                          auth, 
                                          sink_index, 
                                          mappings, 
                                          s.sink_id_col, 
                                          s.sink_capacity_col, 
                                          s.sink_lat_col, 
                                          s.sink_lon_col)

        # Success
        print(f" Successfully indexed {es_success} \n Failed indexed {es_fail}")
        print(f"\n------------------- Sink loaded -------------------\n")
        return sink_out


    # Step . Nodes map
    def call_nodes_map(self, host, auth, source_index, sink_index, query, output_path, cluster_bool, centroids_index):

        s = self.config

        # else:
        df_source = import_data(host, 
                                auth, 
                                source_index, 
                                query, 
                                [s.source_id_col, 
                                s.source_emit_col, 
                                s.source_lat_col, 
                                s.source_lon_col])
        
        # Import sink
        df_sink = import_data(host, 
                              auth, 
                              sink_index, 
                              query, 
                              [s.sink_id_col, 
                              s.sink_capacity_col, 
                              s.sink_lat_col, 
                              s.sink_lon_col])

        # Import centroids
        if cluster_bool == True:

            df_centroids = import_data(host, 
                                       auth, 
                                       centroids_index,
                                       query,
                                       [s.centroid_id_col,
                                       s.centroid_lat_col,
                                       s.centroid_lon_col])
            
            # Compile
            map = nodes_map(df_source,
                            df_sink,
                            s.sink_lat_col,
                            s.sink_lon_col,
                            s.sink_id_col,
                            s.source_lat_col,
                            s.source_lon_col,
                            s.source_id_col,
                            df_centroids,
                            s.centroid_id_col,
                            s.centroid_lat_col,
                            s.centroid_lon_col)

        else:

            # Compile
            map = nodes_map(df_source,
                            df_sink,
                            s.sink_lat_col,
                            s.sink_lon_col,
                            s.sink_id_col,
                            s.source_lat_col,
                            s.source_lon_col,
                            s.source_id_col)
            
        # Export
        map.save(output_path)

        # Success
        print(output_path)
        print(f"\n------------------- Created nodes map -------------------\n")
        return map
    

    # Step . Matrix
    def call_create_matrix(self, host, auth, source_index, sink_index, matrix_index, query, capture_cost, emission_cost, cluster_bool, centroids_index):

        s = self.config
        
        # Import sink
        df_sink = import_data(host, 
                              auth, 
                              sink_index, 
                              query, 
                              [s.sink_id_col, 
                               s.sink_capacity_col, 
                               s.sink_lat_col, 
                               s.sink_lon_col])

        if cluster_bool == True:
            
            # Import source
            df_source = import_data(host, 
                                    auth, 
                                    source_index, 
                                    query, 
                                    [s.source_id_col, 
                                    s.source_emit_col, 
                                    s.source_lat_col, 
                                    s.source_lon_col,
                                    s.source_cluster_col])

            # Centroids
            df_centroids = import_data(host, 
                                       auth, 
                                       centroids_index,
                                       query,
                                       [s.centroid_id_col,
                                       s.centroid_lat_col,
                                       s.centroid_lon_col])
            
            # Compile
            matrix = create_matrix(df_source, 
                                df_sink, 
                                s.source_lat_col, 
                                s.source_lon_col, 
                                s.source_id_col, 
                                s.sink_lat_col, 
                                s.sink_lon_col, 
                                s.sink_id_col, 
                                capture_cost,
                                emission_cost,
                                df_centroids,
                                s.centroid_id_col,
                                s.centroid_lat_col,
                                s.centroid_lon_col,
                                s.source_cluster_col)        
            
        else:

            # Import source
            df_source = import_data(host, 
                                    auth, 
                                    source_index, 
                                    query, 
                                    [s.source_id_col, 
                                    s.source_emit_col, 
                                    s.source_lat_col, 
                                    s.source_lon_col])
            # Compile
            matrix = create_matrix(df_source, 
                                df_sink, 
                                s.source_lat_col, 
                                s.source_lon_col, 
                                s.source_id_col, 
                                s.sink_lat_col, 
                                s.sink_lon_col, 
                                s.sink_id_col, 
                                capture_cost,
                                emission_cost
                                )

        # Export
        es_success, es_fail = matrix_export(matrix, 
                                            host,
                                            auth,
                                            matrix_index)

        # Success
        print(f" Successfully indexed {es_success} \n Failed indexed {es_fail}")
        print(f"\n------------------- Created matrix -------------------\n")
        return matrix
    


    # Step . Network optimization 
    def call_network_opt_min(self, host, auth, source_index, sink_index, matrix_index, query, output_path, cluster_bool, centroids_index):
        
        s = self.config


        # Import sink
        df_sink = import_data(host, 
                              auth, 
                              sink_index, 
                              query, 
                              [s.sink_id_col, 
                              s.sink_capacity_col, 
                              s.sink_lat_col, 
                              s.sink_lon_col])
        
        # Import matrix
        df_matrix = matrix_import(host,
                                auth,
                                matrix_index,
                                query)
        
        if cluster_bool == True:

            # Import source
            df_source = import_data(host, 
                                    auth, 
                                    source_index, 
                                    query, 
                                    [s.source_id_col, 
                                    s.source_emit_col, 
                                    s.source_lat_col, 
                                    s.source_lon_col,
                                    s.source_cluster_col])
            
            # Centroids
            df_centroids = import_data(host, 
                                       auth, 
                                       centroids_index,
                                       query,
                                       [s.centroid_id_col,
                                       s.centroid_lat_col,
                                       s.centroid_lon_col])
            
            network_results = network_opt_min_inter(df_source, 
                                            df_sink, 
                                            df_matrix,
                                            s.source_id_col,
                                            s.source_emit_col,
                                            s.sink_id_col,
                                            s.sink_capacity_col,
                                            s.source_cluster_col)

        else:

            # Import source
            df_source = import_data(host, 
                                    auth, 
                                    source_index, 
                                    query, 
                                    [s.source_id_col, 
                                    s.source_emit_col, 
                                    s.source_lat_col, 
                                    s.source_lon_col])

            # Network optimization
            network_results = network_opt_min(df_source, 
                                            df_sink, 
                                            df_matrix,
                                            s.source_id_col,
                                            s.source_emit_col,
                                            s.sink_id_col,
                                            s.sink_capacity_col)

        # Export
        es_success, es_fail = network_export(network_results, 
                                            host, 
                                            auth, 
                                            s.network_results_index, 
                                            output_path, 
                                            s.source_id_col, 
                                            s.sink_id_col)

        # Success
        print(f" Successfully indexed {es_success} \n Failed indexed {es_fail}")
        print(f"\n------------------- Network loaded -------------------\n")
        return network_results


    # Step . Network map
    def call_network_map(self, host, auth, query, network_index, source_index, sink_index, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon, output_path):
        
        s = self.config

        # Import source
        df_source = import_data(host, 
                                auth, 
                                source_index, 
                                query, 
                                [s.source_id_col, 
                                s.source_emit_col, 
                                s.source_lat_col, 
                                s.source_lon_col])
        
        df_source = df_source.rename(columns={source_id:f'source_{source_id}'})
        df_source[f'source_{source_id}'] = df_source[f'source_{source_id}'].astype(float)

        # Import sink
        df_sink = import_data(host, 
                              auth, 
                              sink_index, 
                              query, 
                              [s.sink_id_col, 
                              s.sink_capacity_col, 
                              s.sink_lat_col, 
                              s.sink_lon_col])
        df_sink = df_sink.rename(columns={sink_id:f'sink_{sink_id}'})

        # Import network
        df_network = network_import(host,
                                 auth,
                                 network_index,
                                 query)
        df_network[f'source_{source_id}'] = df_network[f'source_{source_id}'].astype(float)

        # Create network map
        map = network_map(df_network, df_source, df_sink, source_id, sink_id, source_lat, sink_lat, source_lon, sink_lon)

        # Export
        map.save(output_path)

        # Success
        print(output_path)
        print(f"\n------------------- Created network map -------------------\n")
        return map
    
    # Step . Results calculations
    def call_results_calc(self, host, auth, index, source_id, sink_id, output_path):

        s = self.config

        # Import network
        df_network = network_import(host, auth, index, s.query)
        
        # Compile
        res = results_calc(df_network, source_id, sink_id, s.emission_cost, s.capture_cost)
        
        # Export
        res.to_csv(output_path, index=False)

        #Success
        print(output_path)
        print(f"\n------------------- Calculated results -------------------\n")
        return res