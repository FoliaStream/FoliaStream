import pandas as pd

from .pipe_base import PipelineBase
from pipe.functions.functions_I import create_folder, sink_import, sink_edit, sink_export, source_import, source_edit, source_export, import_data, nodes_map

import warnings
warnings.filterwarnings('ignore')


#/////////////////////////////////////
#           PIPELINE FLOW
#/////////////////////////////////////



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


        # Step . Sink load
        self.call_sink_load(
            in_path = s.in_sink_path,
            host = s.es_host,
            auth = s.es_auth,
            mappings = s.mappings_sink,
            sink_index = s.sink_index
        )


        # Step . Source load
        self.call_source_load(
            api_url = s.source_api_url,
            api_params = { # need to be written in separate yaml and then joined as base-case
                'limit':s.limit,
                'gas':s.gas,
                'countries':s.country,
                'year':s.year,
                'subsectors':s.sector
            },
            host = s.es_host,
            auth = s.es_auth,
            mappings = s.mappings_source,
            source_index = s.source_index
        )

        # Step . Nodes map
        self.call_nodes_map(
            host = s.es_host,
            auth = s.es_auth,
            source_index = s.source_index,
            sink_index = s.sink_index,
            query = s.query,
            output_path = str(str(s.out_fig_path) + str(s.country) + "__" + str(s.year) + "__" + str(s.sector) + "/" + s.nodes_map),
        )


#/////////////////////////////////////
#           CALL FUNCTIONS
#/////////////////////////////////////



    # Step . Create folder
    def call_create_folder(self, 
                           case_paths: list) -> any:

        s = self.config

        # Compile
        for path in case_paths:
            folder = create_folder(str(path))

            print(f"\n{path}")

        # Success
        print(f"\n------------------- Created folders -------------------\n")
        return case_paths, folder
    
    

    # Step . Sink load
    def call_sink_load(self,
                       in_path,
                       host,
                       auth,
                       mappings,
                       sink_index):

        s = self.config

        # Import
        sink_in = sink_import(in_path)

        # Compile
        sink_data = sink_edit(sink_in=sink_in,
                              id_col=s.sink_id_col,
                              country_col=s.sink_country_col,
                              capacity_col=s.sink_capacity_col,
                              lat_col=s.sink_lat_col,
                              lon_col=s.sink_lon_col,
                              country=s.country)

        #Export
        es_success, es_fail = sink_export(sink_data, host, auth, sink_index, mappings, s.sink_id_col)

        # Success
        print(f" Successfully indexed {es_success} \n Failed indexed {es_fail}")
        print(f"\n------------------- Sink loaded -------------------\n")
        return sink_data
    


    # Step . Source load
    def call_source_load(self, api_url, api_params, host, auth, mappings, source_index):

        s = self.config

        # Import
        source_in = source_import(api_url,
                                  api_params)

        # Compile
        source_data = source_edit(source=source_in,
                                  id_col=s.source_id_col,
                                  emit_col=s.source_emit_col,
                                  lat_col=s.source_lat_col,
                                  lon_col=s.source_lon_col)
        
        # Export
        es_success, es_fail = source_export(source=source_data,
                                            host=host,
                                            auth=auth,
                                            index=source_index,
                                            mappings=mappings,
                                            id_col=s.source_id_col)

        # Success
        print(f" Successfully indexed {es_success} \n Failed indexed {es_fail}")
        print(f"\n------------------- Source loaded -------------------\n")
        return source_data



    # Step . Nodes map
    def call_nodes_map(self, host, auth, source_index, sink_index, query, output_path):

        s = self.config

        # Import 
        df_source = import_data(host=host, 
                                auth=auth, 
                                index=source_index, 
                                query=query,
                                columns=[s.source_id_col,
                                         s.source_emit_col,
                                         s.source_lat_col,
                                         s.source_lon_col])
        
        df_sink = import_data(host=host,
                              auth=auth,
                              index=sink_index,
                              query=query,
                              columns=[s.sink_id_col,
                                       s.sink_capacity_col,
                                       s.sink_lat_col,
                                       s.sink_lon_col])
        
        # Compile
        map = nodes_map(source=df_source,
                        sink=df_sink,
                        sink_lat=s.sink_lat_col,
                        sink_lon=s.sink_lon_col,
                        sink_id=s.sink_id_col,
                        source_lat=s.source_lat_col,
                        source_lon=s.source_lon_col,
                        source_id=s.source_id_col)

        # Export
        map.save(output_path)

        # Success
        print(output_path)
        print(f"\n------------------- Created nodes map -------------------\n")
        return map