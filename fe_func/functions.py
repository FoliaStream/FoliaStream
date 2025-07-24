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


def load_store(country):
    df = pd.read_csv('/Users/samuele/Desktop/我/CC/FoliaStream/input/storages.csv')
    country_code = country_name_to_apha3(country)
    df_country = df[df['country'] == country_code]

    df_stats = pd.DataFrame(
        {
            
            'Storage sites': int(len(df_country)),
            'Total storage capacity': int(sum(df_country['sum_mid'])),
            'Offshore': len(df_country[df_country['on_off']=='Offshore'])/len(df_country),
            'Onshore': len(df_country[df_country['on_off']!='Offshore'])/len(df_country),
            'Area': str(df_country['region'].iloc[0]),
            'Share area storage': sum(df_country['sum_mid'])/sum(df[df['region'] == df_country['region'].iloc[0]]['sum_mid'])
         },
         index=[0]
    )


    return df_stats
    

def load_source(country):

    country = country_name_to_apha3(country)

    url = "https://api.climatetrace.org/v6/assets?"
    params = {
        'limit':10000000000000,
        'gas':'co2',
        'countries':country,
        'year':2024
    }

    data = source_import_api(url, params)

    df_source = pd.DataFrame()

    for i in range(len(data)):
        if data[i]['Id'] is not None:
            df_source.at[i,'id'] = data[i]["Id"]
            df_source.at[i,'name'] = data[i]["Name"]
            df_source.at[i,'emission'] = float(data[i]['EmissionsSummary'][0]['EmissionsQuantity'])
            df_source.at[i,'lat'] = float(data[i]['Centroid']['Geometry'][1])
            df_source.at[i,'lon'] = float(data[i]['Centroid']['Geometry'][0])
            df_source.at[i,'sector'] = str(data[i]['Sector'])

    return df_source

