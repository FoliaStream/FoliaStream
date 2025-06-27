import streamlit as st
import pandas as pd

import yaml 
import os
from pipe.streamain import main

# Inputs
options_country = ['Select country','AUS', 'DNK', 'DEU', 'BGD', 'BRA', 'CAN', 'CHN', 'IND', 'IDN', 'JPN', 'MYS', 'MEX', 'NOR', 'PAK', 'KOR', 'LKA', 'GBR', 'USA', 'KAZ', 'KWT', 'MOZ', 'QAT', 'SAU', 'ZAF', 'THA', 'ARE', 'VNM', 'SWE', 'GRC', 'AUT', 'HRV', 'BGR', 'ESP', 'FRA', 'ITA', 'POL', 'CZE', 'SVK', 'HUN', 'IRL', 'ISR', 'MAR', 'DZA', 'ROU', 'NLD']
options_year = ['Select year', 2021, 2022, 2023, 2024] #2020,
options_sector = ['Select sector',"electricity-generation","cement","aluminum","pulp-and-paper","chemicals","oil-and-gas-refining","coal-mining","bauxite-mining","iron-mining","copper-mining"] #"domestic-aviation","international-aviation","net-forest-land","net-wetland","net-shrubgrass","cropland-fires"
options_transport = ['Select transport','pipe', 'truck_ship']
options_network = ['Select network type', 'Direct connection', 'Dijkstra', '1k-cluster']

country = st.selectbox("Country", options=options_country)
year = st.selectbox("Year", options=options_year)
sector = st.selectbox("Sector", options=options_sector)
capture_cost = st.number_input("Capture cost", step=1, value=0)
emission_cost = st.number_input("Emission cost", step=1, value=0)
transport_method = st.selectbox("Transport method", options=options_transport)
network_type = st.selectbox("Network type", options=options_network)

# hub_cost 

# Give option input also transport cost 

with st.sidebar:
    st.image(f"{os.getcwd()}/logo.png")

if country != 'Select country' and year != 'Select year' and sector != 'Select sector' and transport_method != 'Select transport' and network_type != 'Select network type':

    if st.button("RUN"):
        with open(f'{os.getcwd()}/pipe/config/case.yaml', "w") as f:

            data = {
                "country" : country,
                "year" : int(year),
                "sector" : sector,
                "capture_cost" : capture_cost,
                "emission_cost" : emission_cost,
                "transport_method" : transport_method,
                "network_type": network_type
            }
            yaml.dump(data, f, default_flow_style=False)

        main()

        # Outputs
        st.dataframe(pd.read_csv(str(f"{os.getcwd()}/output/final/csv/{country}__{year}__{sector}/network_results.csv"))[['source_id', 'sink_id', 'co2_transported']], 
                     use_container_width=True,
                     column_config={
                        "sink_id":st.column_config.TextColumn("Sink ID"),
                        "source_id":st.column_config.TextColumn("Source ID"),
                        "co2_transported":st.column_config.NumberColumn("CO2")
                     },
                     hide_index=True)

        map = open(str(f"{os.getcwd()}/output/final/fig/{country}__{year}__{sector}/network_map_out.html"))
        st.components.v1.html(map.read(), height=500, scrolling=True)

else:
    st.button("RUN", disabled=True)
