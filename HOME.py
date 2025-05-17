import streamlit as st
import pandas as pd

import yaml 
import os
from pipe.streamain import main

options_country = ['AUS', 'DNK', 'DEU', 'BGD', 'BRA', 'CAN', 'CHN', 'IND', 'IDN', 'JPN', 'MYS', 'MEX', 'NOR', 'PAK', 'KOR', 'LKA', 'GBR', 'USA', 'KAZ', 'KWT', 'MOZ', 'QAT', 'SAU', 'ZAF', 'THA', 'ARE', 'VNM', 'SWE', 'GRC', 'AUT', 'HRV', 'BGR', 'ESP', 'FRA', 'ITA', 'POL', 'CZE', 'SVK', 'HUN', 'IRL', 'ISR', 'MAR', 'DZA', 'ROU', 'NLD']
options_year = [2020, 2021, 2022, 2023, 2024]
options_sector = ["electricity-generation","cement","aluminum","pulp-and-paper","chemicals","domestic-aviation","international-aviation","oil-and-gas-refining","coal-mining","bauxite-mining","iron-mining","copper-mining","net-forest-land","net-wetland","net-shrubgrass","cropland-fires"]

country = st.selectbox("Country", options=options_country)
year = st.selectbox("Year", options=options_year, index = 4)
sector = st.selectbox("Sector", options=options_sector)
capture_cost = st.number_input("Capture cost", step=1)
emission_cost = st.number_input("Emission cost", step=1)

if st.button("RUN"):
    with open(f'{os.getcwd()}/pipe/config/case.yaml', "w") as f:

        data = {

            "country" : country,
            "year" : int(year),
            "sector" : sector,
            "capture_cost" : capture_cost,
            "emission_cost" : emission_cost
        }
        yaml.dump(data, f, default_flow_style=False)

    main()

    st.dataframe(pd.read_csv(f"{os.getcwd()}/output/temp/csv/{country}__{year}__{sector}/sink_raw.csv"))