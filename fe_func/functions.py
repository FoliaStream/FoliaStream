import pandas as pd 
import requests
import pycountry
from pipe.functions.functions_I import source_import_api




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
    

def load_geojson():
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    return requests.get(url).json()


def country_name_to_apha3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None 
    
def alpha3_to_country_name(alpha3):
    try:
        country = pycountry.countries.get(alpha_3=alpha3.upper())
        if country:
            return country.name
        else:
            return None
    except KeyError:
        return None


# def load_store(country):
#     df = pd.read_csv('/Users/samuele/Desktop/我/CC/FoliaStream/input/storages.csv')
#     country_code = country_name_to_apha3(country)
#     df_country = df[df['country'] == country_code]

#     df_stats = pd.DataFrame(
#         {
            
#             'Storage sites': int(len(df_country)),
#             'Total storage capacity': int(sum(df_country['sum_mid'])),
#             'Offshore': len(df_country[df_country['on_off']=='Offshore'])/len(df_country),
#             'Onshore': len(df_country[df_country['on_off']!='Offshore'])/len(df_country),
#             'Area': str(df_country['region'].iloc[0]),
#             'Share area storage': sum(df_country['sum_mid'])/sum(df[df['region'] == df_country['region'].iloc[0]]['sum_mid'])
#          },
#          index=[0]
#     )
#     breakpoint()


#     return df_stats

def load_store(country):
    df = pd.read_csv('/Users/samuele/Desktop/我/CC/FoliaStream/input/storages.csv')
    country_code = country_name_to_apha3(country)
    df_country = df[df['country'] == country_code]
    
    # Calculate onshore/offshore counts and capacities
    offshore_sites = df_country[df_country['on_off'] == 'Offshore']
    onshore_sites = df_country[df_country['on_off'] != 'Offshore']
    
    offshore_count = len(offshore_sites)
    onshore_count = len(onshore_sites)
    
    offshore_capacity = sum(offshore_sites['sum_mid'])
    onshore_capacity = sum(onshore_sites['sum_mid'])
    total_capacity = offshore_capacity + onshore_capacity
    
    df_stats = pd.DataFrame(
        {
            'Storage sites': len(df_country),
            'Total storage capacity': total_capacity,
            'Offshore count': offshore_count,
            'Onshore count': onshore_count,
            'Offshore capacity': offshore_capacity,
            'Onshore capacity': onshore_capacity,
            'Offshore %': offshore_count/len(df_country),
            'Onshore %': onshore_count/len(df_country),
            'Area': str(df_country['region'].iloc[0]),
            'Share area storage': total_capacity/sum(df[df['region'] == df_country['region'].iloc[0]]['sum_mid'])
        },
        index=[0]
    )


    return df_stats, df_country
    

# def load_source(country, years):

#     country = country_name_to_apha3(country)

#     url = "https://api.climatetrace.org/v6/assets?"

#     if len(years) == 1:
#         params = {
#             'limit':10000000000000,
#             'gas':'co2',
#             'countries':country,
#             'year':years[0]
#         }

#         data = source_import_api(url, params)

#         df_source = pd.DataFrame()

#         for i in range(len(data)):
#             if data[i]['Id'] is not None:
#                 df_source.at[i,'id'] = data[i]["Id"]
#                 df_source.at[i,'name'] = data[i]["Name"]
#                 df_source.at[i,'emission'] = float(data[i]['EmissionsSummary'][0]['EmissionsQuantity'])
#                 df_source.at[i,'lat'] = float(data[i]['Centroid']['Geometry'][1])
#                 df_source.at[i,'lon'] = float(data[i]['Centroid']['Geometry'][0])
#                 df_source.at[i,'sector'] = str(data[i]['Sector'])

#         return df_source

#     elif len(years) >= 1:

#         for year in years:
#             params = {
#             'limit':10000000000000,
#             'gas':'co2',
#             'countries':country,
#             'year':year
#         }

#         data = source_import_api(url, params)




def load_source(country, years):
    """
    Fetches emissions data from ClimateTrace API for a country across specified years.
    Returns a DataFrame with emissions columns named by actual year (e.g., emissions_2024).
    
    Args:
        country (str): Country name (will be converted to Alpha-3 code)
        years (list): List of years (e.g., [2024, 2025, 2026])
    
    Returns:
        pd.DataFrame: Columns: [id, name, lat, lon, sector, emissions_2024, emissions_2025, ...]
    """
    country = country_name_to_apha3(country)
    url = "https://api.climatetrace.org/v6/assets?"
    
    master_df = pd.DataFrame()
    asset_id_to_index = {}  # Maps asset IDs to DataFrame indices
    current_index = 0
    
    for year in sorted(years):  # Process years in order
        params = {
            'limit': 10000000000000,
            'gas': 'co2',
            'countries': country,
            'year': year
        }
        
        data = source_import_api(url, params)
        
        for asset in data:
            if asset['Id'] is None:
                continue
                
            asset_id = asset["Id"]
            
            # Add new asset to DataFrame if not already present
            if asset_id not in asset_id_to_index:
                master_df.at[current_index, 'id'] = asset_id
                master_df.at[current_index, 'name'] = asset["Name"]
                master_df.at[current_index, 'lat'] = float(asset['Centroid']['Geometry'][1])
                master_df.at[current_index, 'lon'] = float(asset['Centroid']['Geometry'][0])
                master_df.at[current_index, 'sector'] = str(asset['Sector'])
                asset_id_to_index[asset_id] = current_index
                current_index += 1
            
            # Add emissions for this year
            row_idx = asset_id_to_index[asset_id]
            emissions = float(asset['EmissionsSummary'][0]['EmissionsQuantity'])
            master_df.at[row_idx, f'emissions_{year}'] = emissions
    
    # Ensure consistent column order: metadata first, then emissions by year
    metadata_cols = ['id', 'name', 'lat', 'lon', 'sector']
    emission_cols = sorted([col for col in master_df.columns if col.startswith('emissions_')])
    master_df = master_df[metadata_cols + emission_cols]
    
    return master_df