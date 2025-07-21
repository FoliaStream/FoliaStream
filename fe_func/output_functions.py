import pandas as pd 

def flow_table(source_path, sink_path, output_path):

    df_source = pd.read_csv(source_path)
    df_sink = pd.read_csv(sink_path)
    df_output = pd.read_csv(output_path)[['source_id', 'sink_id', 'co2_transported']]

    df_output['sink_id'] = df_output['sink_id'].astype(str)
    df_sink['id'] = df_sink['id'].astype(str)

    df_output['source_name'] = pd.Series()
    df_output['sink_name'] = pd.Series()
    for i, row in df_output.iterrows():

        df_output.at[i, 'source_name'] = str(df_source[df_source['id'] == row['source_id']]['name'].values[0])
        if row['sink_id'] == "Atmosphere":
            pass
        else:
            df_output.at[i,'sink_name'] = str(df_sink[df_sink['id'] == row['sink_id']]['site_name'].values[0])
    
    df_output = df_output[['source_name','source_id', 'sink_name', 'sink_id', 'co2_transported']]

    return df_output
    


def cost_table(flow_output, capture_cost, emission_cost):

    output = {
        'co2_captured':flow_output[flow_output['sink_id'] != 'Atmosphere']['co2_transported'].sum(),
        'co2_emitted':flow_output[flow_output['sink_id'] == 'Atmosphere']['co2_transported'].sum(),
        'tot_capture_cost':flow_output[flow_output['sink_id'] != 'Atmosphere']['co2_transported'].sum() * capture_cost,
        'tot_emission_cost':flow_output[flow_output['sink_id'] == 'Atmosphere']['co2_transported'].sum() * emission_cost
    }

    df_output = pd.DataFrame.from_dict(output, orient='index').T

    return df_output