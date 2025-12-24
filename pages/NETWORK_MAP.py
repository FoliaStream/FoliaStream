import pandas as pd
import streamlit as st 
import plotly.graph_objects as go

import yaml
import os 

from streamlit_plotly_events import plotly_events
from pipe.streamain import main
from fe_func.functions import load_geojson, flow_table, country_name_to_apha3

# --- PAGE CONFIG ---

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
highlighted_countries_names = [
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

highlighted_countries_codes = [country_name_to_apha3(country) for country in highlighted_countries_names]

# Create figure

# Colors
HIGHLIGHT_COLOR = "rgba(0, 255, 0, 0.5)"      
BORDER_COLOR = "rgb(0, 80, 200)"          
BACKGROUND_COLOR = "rgb(245, 248, 250)"   
OCEAN_COLOR = "rgb(220, 235, 255)"  
COUNTRY_COLOR = "rgba(0, 80, 200, 0.4)"

# Globe
globe = go.Figure()

# Hover texts
hover_texts = []
for country in highlighted_countries_names:
    hover_texts.append(f"<span style='font-size: 20px;'>{country}</span><b></b><br>")

# Add highlighted countries
globe.add_trace(go.Choropleth(
    locations=highlighted_countries_codes,
    z=[1] * len(highlighted_countries_codes),
    colorscale=[[0, HIGHLIGHT_COLOR], [1,HIGHLIGHT_COLOR]],
    showscale=False,
    marker_line_color=BORDER_COLOR,
    marker_line_width=1.5,
    geo='geo',
    text=hover_texts,
    hoverinfo='text',
    hovertemplate='%{text}<extra></extra>',
    name='GLOBE',
    customdata=highlighted_countries_names
))

# Add features
globe.update_layout(
    geo=dict(
        showland=True,
        landcolor=BACKGROUND_COLOR,
        showocean=True,
        oceancolor=OCEAN_COLOR,
        showcountries=True,
        countrycolor=COUNTRY_COLOR,
        countrywidth=1.5,
        showcoastlines=True,
        coastlinecolor=BORDER_COLOR,
        projection=dict(
            type='orthographic',
            scale=0.9,
            rotation=dict(lon=50, lat=21, roll=0)
        ),
        bgcolor='rgba(0, 0, 0, 0)',
        showframe=False,
        lataxis=dict(showgrid=False),
        lonaxis=dict(showgrid=False)
    ),
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, t=0, b=0),
    dragmode='pan',
    autosize=True
)

select_points = plotly_events(
    globe, 
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=800, 
    key="globe"
)



# Country selected -> Network params
if select_points:
    clicked_point = select_points[0]
    country_index = clicked_point.get('pointIndex', 0)
    selected_country = highlighted_countries_names[country_index]

    st.markdown(f"<h1 style='text-align: center;'>{selected_country}</h1>", unsafe_allow_html=True)
    st.divider()

    with st.spinner("Work in progress..."):

        if selected_country in highlighted_countries_names:

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
                    map = open(str(f"{os.getcwd()}/output/final/fig/{country}__{year}__{sector}/network_map_out.html"))
                    st.components.v1.html(map.read(), height=500, scrolling=True)

            else:
                st.button("RUN", disabled=True)

        else: 
            st.write("No data available")

                


<<<<<<< HEAD
                st.subheader("Network Map:")

                map_network = open(str(f"{os.getcwd()}/output/final/fig/{country}__{year}__{sector}/network_map_out.html"))
                st.components.v1.html(map_network.read(),height=500, width=15000, scrolling=True)

        else:
            st.button("RUN", disabled=True)

    else: 
        st.write("No data available")
=======
>>>>>>> devop
