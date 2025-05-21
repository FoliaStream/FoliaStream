import streamlit as st
import pandas as pd

import yaml 
import os
from pipe.streamain import main

# Inputs
options_country = ['Select country','AUS', 'DNK', 'DEU', 'BGD', 'BRA', 'CAN', 'CHN', 'IND', 'IDN', 'JPN', 'MYS', 'MEX', 'NOR', 'PAK', 'KOR', 'LKA', 'GBR', 'USA', 'KAZ', 'KWT', 'MOZ', 'QAT', 'SAU', 'ZAF', 'THA', 'ARE', 'VNM', 'SWE', 'GRC', 'AUT', 'HRV', 'BGR', 'ESP', 'FRA', 'ITA', 'POL', 'CZE', 'SVK', 'HUN', 'IRL', 'ISR', 'MAR', 'DZA', 'ROU', 'NLD']
options_year = ['Select year', 2020, 2021, 2022, 2023, 2024]
options_sector = ['Select sector',"electricity-generation","cement","aluminum","pulp-and-paper","chemicals","domestic-aviation","international-aviation","oil-and-gas-refining","coal-mining","bauxite-mining","iron-mining","copper-mining","net-forest-land","net-wetland","net-shrubgrass","cropland-fires"]
options_transport = ['Select transport','pipe', 'truck_ship']

country = st.selectbox("Country", options=options_country)
year = st.selectbox("Year", options=options_year)
sector = st.selectbox("Sector", options=options_sector)
capture_cost = st.number_input("Capture cost", step=1, value=0)
emission_cost = st.number_input("Emission cost", step=1, value=0)
transport_method = st.selectbox("Transport method", options=options_transport)
# Give option input also transport cost 

with st.sidebar:
    st.image(f"{os.getcwd()}/logo.png")

if country != 'Select country' and year != 'Select year' and sector != 'Select sector' and transport_method != 'Select transport':

    if st.button("RUN"):
        with open(f'{os.getcwd()}/pipe/config/case.yaml', "w") as f:

            data = {

                "country" : country,
                "year" : int(year),
                "sector" : sector,
                "capture_cost" : capture_cost,
                "emission_cost" : emission_cost,
                "transport_method" : transport_method
            }
            yaml.dump(data, f, default_flow_style=False)

        main()

        st.dataframe(pd.read_csv(str(f"{os.getcwd()}/output/final/csv/{country}__{year}__{sector}/network_results.csv")))
        map = open(str(f"{os.getcwd()}/output/final/fig/{country}__{year}__{sector}/network_map_out.html"))
        st.components.v1.html(map.read(), height=500, scrolling=True)

else:
    st.button("RUN", disabled=True)
