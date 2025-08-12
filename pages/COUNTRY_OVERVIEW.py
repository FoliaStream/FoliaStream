import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


import os
import folium


from streamlit_folium import st_folium
from pipe.streamain import main
from fe_func.functions import load_geojson, load_store, load_source




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



# --- STATISTICS ---

if clicked and clicked.get("last_active_drawing") is not None:
    
    selected_country = clicked["last_active_drawing"]["properties"]["name"]
    st.markdown(f"<h1 style='text-align: center;'>{selected_country}</h1>", unsafe_allow_html=True)
    st.divider()

    if selected_country in highlighted_countries:

        # Side-by-side layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("**Sources**")
        with col2: 
            st.header("**Sinks**")

        # SOURCE
        with col1:

            selected_year = st.selectbox('Reference Year',options=[2024,2023,2022,2021],index=0)
            
            df_source = load_source(selected_country, selected_year)
            st.metric(f"Total number of industrial sites in {selected_country} (limit 10,000)", f"{int(len(df_source)):,}")
            st.metric(f"Total emissions in {selected_country}", f"{int(sum(df_source['emission'])/1000000):,} Mt")

            sectors_list = list(df_source['sector'].unique())

            # Get sector counts
            sector_counts = df_source['sector'].value_counts()

            # Set a threshold for what's considered "small" 
            threshold = 50 

            # Separate large and small sectors
            main_sectors = sector_counts[sector_counts >= threshold]
            other_sectors = sector_counts[sector_counts < threshold]

            # Combine the small sectors into "Others"
            if len(other_sectors) > 0:
                sectors_list = main_sectors.index.tolist() + ['Others']
                sectors_count = main_sectors.tolist() + [other_sectors.sum()]
            else:
                sectors_list = main_sectors.index.tolist()
                sectors_count = main_sectors.tolist()

            for sect in sectors_list:
                sectors_count.append(int(len(df_source[df_source['sector'] == sect])))

            sectors_pie = go.Figure(data=[go.Pie(
                labels=sectors_list,
                values=sectors_count,
                hole =.6,
                hoverinfo='label+percent',
                textinfo='value'
            )])

            sectors_pie.update_layout(
                title="Number of Industrial Sites per Sector",
                showlegend=True,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.markdown('###')

            st.plotly_chart(sectors_pie, use_container_width=True)

        # SINK
        with col2:

            df_store = load_store(selected_country)

            st.metric("Area", f"{str(df_store['Area'].iloc[0])}")
            st.metric(f"Total number of potential storage sites in {selected_country}", f"{int(df_store['Storage sites']):,}")
            st.metric(f"Total storage capacity in {selected_country}", f"{int(df_store['Total storage capacity']/1_000_000):,} Mt")
            
            # Right column: donut chart with green colors
            colors = ['#2ecc71', '#27ae60']  # Two green shades

            on_off = go.Figure(data=[go.Pie(
                labels=['Onshore', 'Offshore'],
                values=[round(float(df_store['Onshore']),2)*100, round(float(df_store['Offshore']),2)*100],
                hole=.6,
                hoverinfo='label+percent',
                textinfo='value',
                marker=dict(colors=colors),
                texttemplate='%{value}%'
            )])

            on_off.update_layout(
                title="Onshore vs Offshore CO₂ Storage Sites",
                showlegend=True,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.markdown('###')

            st.plotly_chart(on_off, use_container_width=True)

    else: 
        st.write("No data available")