import streamlit as st
import pandas as pd

import yaml 
import os
from pipe.streamain import main
from fe_func.output_functions import flow_table, cost_table

# Inputs
options_country = ['Select country','AUS', 'DNK', 'DEU', 'BGD', 'BRA', 'CAN', 'CHN', 'IND', 'IDN', 'JPN', 'MYS', 'MEX', 'NOR', 'PAK', 'KOR', 'LKA', 'GBR', 'USA', 'KAZ', 'KWT', 'MOZ', 'QAT', 'SAU', 'ZAF', 'THA', 'ARE', 'VNM', 'SWE', 'GRC', 'AUT', 'HRV', 'BGR', 'ESP', 'FRA', 'ITA', 'POL', 'CZE', 'SVK', 'HUN', 'IRL', 'ISR', 'MAR', 'DZA', 'ROU', 'NLD']
options_year = ['Select year', 2021, 2022, 2023, 2024] #2020,
options_sector = ['Select sector',"electricity-generation","cement","aluminum","pulp-and-paper","chemicals","oil-and-gas-refining","coal-mining","bauxite-mining","iron-mining","copper-mining"] #"domestic-aviation","international-aviation","net-forest-land","net-wetland","net-shrubgrass","cropland-fires"
options_transport = ['Select transport','pipe', 'truck_ship']
options_network = ['Select network type', 'Direct connection', 'Dijkstra connection', '1k-cluster connection']
options_capture = ['Select capture method', 'Carbon Capture (CC)', 'Direct Air Capture (DAC)']

country = st.selectbox("Country", options=options_country)
year = st.selectbox("Year", options=options_year)
sector = st.selectbox("Sector", options=options_sector)
capture_cost = st.number_input("Capture cost", step=1, value=0)
emission_cost = st.number_input("Emission cost", step=1, value=0)
transport_method = st.selectbox("Transport method", options=options_transport)
network_type = st.selectbox("Network type", options=options_network)
capture_method = st.selectbox("Capture method", options_capture)



# Give option input also transport cost 

with st.sidebar:
    st.image(f"{os.getcwd()}/logo.png")

if country != 'Select country' and year != 'Select year' and sector != 'Select sector' and transport_method != 'Select transport' and network_type != 'Select network type' and capture_method != 'Select capture method':

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

        cost_results = cost_table(flow_results, capture_cost, emission_cost)

        # Flow results
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

        # Cost results
        st.dataframe(cost_results,
                     use_container_width=True,
                     column_config={
                         'co2_captured':st.column_config.NumberColumn("Captured CO2 (ton)"),
                         'co2_emitted':st.column_config.NumberColumn("Emitted CO2 (ton)"),
                         'tot_capture_cost':st.column_config.NumberColumn("Total cost of capture ($)"),
                         'tot_emission_cost':st.column_config.NumberColumn("Total cost of emission ($)")
                     },
                     hide_index=True)

        map = open(str(f"{os.getcwd()}/output/final/fig/{country}__{year}__{sector}/network_map_out.html"))
        st.components.v1.html(map.read(), height=500, scrolling=True)


else:
    st.button("RUN", disabled=True)
