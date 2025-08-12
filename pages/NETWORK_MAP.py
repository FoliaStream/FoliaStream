import streamlit as st
import pandas as pd


import yaml 
import os
import folium


from streamlit_folium import st_folium
from pipe.streamain import main
from fe_func.functions import load_geojson, flow_table, country_name_to_apha3



# --- SIDEBAR ---

st.set_page_config(page_title="FoliaStream - CO₂ Network", layout="wide")

with st.sidebar:
    st.image(f"{os.getcwd()}/logo.png")
    st.divider()
    st.subheader("DATA SOURCES:")
    st.link_button("Climate TRACE", "https://climatetrace.org/", use_container_width=True)
    st.link_button("Oil & Gas Climate Initiative (OGCI)", "https://www.ogci.com/", use_container_width=True)



# --- HEADER ---

st.header("Select country:")
st.markdown('#')



# --- MAP ---

# Available countries
highlighted_countries = [
    'Algeria',
    'Australia',
    'Austria',
    'Bangladesh',
    'Brazil',
    'Bulgaria',
    'Canada',
    'China',
    'Croatia',
    'Czech Republic',
    'Denmark',
    'France',
    'Germany',
    'Greece',
    'Hungary',
    'India',
    'Indonesia',
    'Ireland',
    'Israel',
    'Italy',
    'Japan',
    'Kazakhstan',
    'Malaysia',
    'Mexico',
    'Morocco',
    'Mozambique',
    'Netherlands',
    'Norway',
    'Pakistan',
    'Poland',
    'Qatar',
    'Romania',
    'Saudi Arabia',
    'Slovakia',
    'South Africa',
    'South Korea',
    'Spain',
    'Sri Lanka',
    'Sweden',
    'Thailand',
    'United Kingdom',
    'United States of America',
    'Vietnam'
]


# Load json maps
geojson_data = load_geojson() 

# Country Selection Map
map = folium.Map(location=[20, 0], zoom_start=2, min_zoom=2)

folium.GeoJson(
    geojson_data,
    name="Countries",
    style_function=lambda feature: {
        "fillColor": "#00ff00" if feature["properties"]["name"] in highlighted_countries else "#3388ff",
        "color": "black" if feature["properties"]["name"] in highlighted_countries else "grey",
        "weight": 1.5 if feature["properties"]["name"] in highlighted_countries else 0.5,
        "fillOpacity": 0.5 if feature["properties"]["name"] in highlighted_countries else 0.1,
    },
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Country:"]),
).add_to(map)

clicked = st_folium(map, width='100%', height=600)

if clicked and clicked.get("last_active_drawing") is not None:

    selected_country = clicked["last_active_drawing"]["properties"]["name"]
    st.markdown(f"<h1 style='text-align: center;'>{selected_country}</h1>", unsafe_allow_html=True)
    st.divider()

    if selected_country in highlighted_countries:

        options_year = ['Select year', 2021, 2022, 2023, 2024] 
        options_sector = ['Select sector',"electricity-generation","cement","aluminum","pulp-and-paper","chemicals","oil-and-gas-refining","coal-mining","bauxite-mining","iron-mining","copper-mining"] 
        options_transport = ['Select transport','pipe', 'truck_ship']
        options_network = ['Select network type', 'Direct connection', 'Dijkstra connection', '1k-cluster connection']
        options_capture = ['Select capture method', 'Carbon Capture (CC)', 'Direct Air Capture (DAC)']

        year = st.selectbox("Year", options=options_year)
        sector = st.selectbox("Sector", options=options_sector)
        capture_cost = st.number_input("Capture cost", step=1, value=0, min_value=0)
        emission_cost = st.number_input("Emission cost", step=1, value=0, min_value=0)
        transport_method = st.selectbox("Transport method", options=options_transport)
        network_type = st.selectbox("Network type", options=options_network)
        capture_method = st.selectbox("Capture method", options_capture)

        if year != 'Select year' and sector != 'Select sector' and transport_method != 'Select transport' and network_type != 'Select network type' and capture_method != 'Select capture method':
            
            country = country_name_to_apha3(selected_country)
            if st.button("RUN"):

                with open(f'{os.getcwd()}/pipe/config/case.yaml', "w") as f:

                    data = {
                        "country" : country,
                        "year" : int(year),
                        "sector" : sector,
                        "capture_cost" : capture_cost,
                        "emission_cost" : emission_cost,
                        "transport_method" : transport_method,
                        "network_type": network_type,
                        "capture_method":capture_method
                    }
                    yaml.dump(data, f, default_flow_style=False)

                with st.spinner("Work in progress..."):
                    main()

                # Outputs prep
                if capture_method == 'Carbon Capture (CC)':

                    flow_results = flow_table(f"{os.getcwd()}//output/temp/csv/{country}__{year}__{sector}/source_raw.csv", 
                                            f"{os.getcwd()}//output/temp/csv/{country}__{year}__{sector}/sink_raw.csv",
                                            f"{os.getcwd()}/output/final/csv/{country}__{year}__{sector}/network_results.csv")
                
                elif capture_method == 'Direct Air Capture (DAC)':

                    flow_results = flow_table(f"{os.getcwd()}//output/temp/csv/{country}__{year}__{sector}/dac.csv", 
                                            f"{os.getcwd()}//output/temp/csv/{country}__{year}__{sector}/sink_raw.csv",
                                            f"{os.getcwd()}/output/final/csv/{country}__{year}__{sector}/network_results.csv")


                # Flow results
                st.markdown("<hr>", unsafe_allow_html=True)
                st.subheader("Flows:")

                st.dataframe(flow_results,
                            use_container_width=True,
                            column_config={
                                "sink_id":st.column_config.TextColumn("Sink ID"),
                                "source_id":st.column_config.TextColumn("Source ID"),
                                "co2_transported":st.column_config.NumberColumn("CO2 (ton)"),
                                "source_name":st.column_config.TextColumn("Source Name"),
                                "sink_name":st.column_config.TextColumn("Sink Name")
                            },
                            hide_index=True)
                

                # Totals
                st.subheader("Overview:")
                st.dataframe(pd.read_csv(f"{os.getcwd()}//output/temp/csv/{country}__{year}__{sector}/totals.csv"),
                            use_container_width=True,
                            column_config={
                                "Captured CO2":st.column_config.NumberColumn("Captured CO2 (ton)"),
                                "Transport Cost":st.column_config.NumberColumn("Transport Cost (€)"),
                                "Capture Cost":st.column_config.NumberColumn("Capture Cost (€)"),
                                "Storage Cost":st.column_config.NumberColumn("Storage Cost (€)"),
                                "Emitted CO2":st.column_config.NumberColumn("Emitted CO2 (ton)"),
                                "Emission Cost":st.column_config.NumberColumn("Emission Cost (€)")
                            },
                            hide_index=True)

                st.subheader("Network Map:")
                st.markdown("""
                <style>
                    div[data-testid="stHorizontalBlock"] {
                        width: 100% !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                map_network = open(str(f"{os.getcwd()}/output/final/fig/{country}__{year}__{sector}/network_map_out.html"))
                st.components.v1.html(map_network.read(),height=500, scrolling=True, width=None)

        else:
            st.button("RUN", disabled=True)

    else: 
        st.write("No data available")