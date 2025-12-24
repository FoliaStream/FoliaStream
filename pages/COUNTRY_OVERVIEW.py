import pandas as pd
import streamlit as st 
import plotly.graph_objects as go 
import plotly.express as px


import os

from streamlit_plotly_events import plotly_events
from pipe.streamain import main
from fe_func.functions import load_store, load_source, country_name_to_apha3
from pipe.pipe_flow.pipe_flow import nodes_map


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
    # 'Sri Lanka',
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
    colorscale=[[0, HIGHLIGHT_COLOR], [1, HIGHLIGHT_COLOR]],
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


globe.update_layout(
    geo=dict(
        showland=True,
        landcolor='rgb(245, 248, 250)',
        showocean=True,
        oceancolor='rgb(220, 235, 255)',
        showcountries=True,
        countrycolor='rgba(0, 80, 200, 0.4)',
        countrywidth = 1.5,
        showcoastlines=True,
        coastlinecolor='rgb(0, 80, 200)',
        projection=dict(
            type='orthographic',
            scale=0.9,
            rotation=dict(lon=50, lat=21, roll=0)),
        bgcolor='rgba(0, 0, 0, 0)',
        showframe=False,
        lataxis=dict(showgrid=False),
        lonaxis=dict(showgrid=False),
    ),
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, t=0, b=0),
    # width=800,
    dragmode='pan',
    autosize=True
)



selected_points = plotly_events(
    globe, 
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=800,  
    key="globe",
)



if selected_points:
    clicked_point = selected_points[0]
    country_index = clicked_point.get('pointIndex', 0)
    selected_country = highlighted_countries_names[country_index]
    
    st.markdown(f"<h1 style='text-align: center;'>{selected_country}</h1>", unsafe_allow_html=True)
    st.divider()

    with st.spinner("Work in progress..."):
            
        if selected_country in highlighted_countries_names:
            

            # Side-by-side layout
            col1, col2 = st.columns([1, 1])

            with col1:
                st.header("**Sources**")
            with col2: 
                st.header("**Sinks**")

            # SOURCE
            with col1:

                options_year = [2021, 2022, 2023, 2024]
                selected_year = st.selectbox('Reference Year', options=options_year, index=3)

                # Filter out zero-emission sites before any calculations
                df_source = load_source(selected_country, [selected_year])
                df_source = df_source[df_source[f'emissions_{selected_year}'] > 0]  # THIS IS THE KEY FILTER

                # Metrics (now only counting sites with emissions)
                st.metric(f"Total number of emitting sites in {selected_country}", f"{int(len(df_source)):,}")
                st.metric(f"Total emissions in {selected_country}", f"{int(sum(df_source[f'emissions_{selected_year}'])/1000000):,} Mt")
                

                # Pie chart - Sectors (only for sectors with emissions)
                sector_counts = df_source['sector'].value_counts()

                # Group small sectors into "Others"
                threshold = 10
                main_sectors = sector_counts[sector_counts >= threshold]
                other_sectors = sector_counts[sector_counts < threshold]

                if len(other_sectors) > 0:
                    sectors_list = main_sectors.index.tolist() + ['Others']
                    sectors_count = main_sectors.tolist() + [other_sectors.sum()]
                else:
                    sectors_list = main_sectors.index.tolist()
                    sectors_count = main_sectors.tolist()

                # Create the pie chart
                sectors_pie = go.Figure(data=[go.Pie(
                    labels=sectors_list,
                    values=sectors_count,
                    hole=.6,
                    hoverinfo='label+percent+value',
                    textinfo='value'
                )])

                sectors_pie.update_layout(
                    title=f"Industrial Sites per Sector ({selected_year})",
                    showlegend=True,
                    margin=dict(t=50, b=0, l=0, r=0)
                )

                st.markdown('###')
                st.plotly_chart(sectors_pie, use_container_width=True)


                # Line graph --> time x emissions x sector
                df_source_all_year = load_source(selected_country, options_year)
                
                plot_data = []
                for year in options_year:
                    year_col = f'emissions_{year}'
                    year_df = df_source_all_year[['sector', year_col]].copy()
                    year_df['year'] = year
                    year_df = year_df.rename(columns={year_col: 'emissions'})
                    year_df = year_df.groupby(['sector', 'year'])['emissions'].sum().reset_index()
                    plot_data.append(year_df)

                plot_df = pd.concat(plot_data)
                plot_df = plot_df[plot_df['emissions'] > 0]


                fig = px.area(
                    plot_df,
                    x='year',
                    y='emissions',
                    color='sector',
                    title='CO₂ Emissions by Sector',
                    labels={'emissions': 'Emissions (metric tons)'},
                    category_orders={"year": [2021, 2022, 2023, 2024]},
                    hover_data={
                        'year': True,
                        'sector': True,
                        'emissions': ':.1f'  # Format with 1 decimal place
                    }
                )

                # 4. Customize tooltip appearance
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"  # Sector name (bold)
                        "Year: %{x}<br>"               # Year
                        "Emissions: %{y:,.1f} tons<br>"  # Formatted value
                        "<extra></extra>"              # Remove secondary box
                    )
                )
        
                fig.update_xaxes(
                                type='category',
                                tickmode='array',
                                tickvals=options_year,
                                ticktext=options_year
                            )

                st.plotly_chart(fig, use_container_width=True)

        

            # SINK
            with col2:
                
                df_store, df_sink = load_store(selected_country)
                
                # Convert capacities to Mt (million tons)
                total_capacity_mt = float(df_store['Total storage capacity'])/1_000_000
                onshore_capacity_mt = float(df_store['Onshore capacity'])/1_000_000
                offshore_capacity_mt = float(df_store['Offshore capacity'])/1_000_000
                
                # Display metrics
                st.metric("Area", f"{str(df_store['Area'].iloc[0])}")
                st.metric(f"Total storage sites in {selected_country}", 
                        f"{int(df_store['Storage sites']):,}")
                st.metric(f"Total storage capacity", 
                        f"{total_capacity_mt:,.1f} Mt")
                
                colors = ['#2ecc71', '#27ae60']  # Green shades

                # 1. Donut Chart - Site Distribution
                st.markdown('###')
                site_distribution = go.Figure(data=[go.Pie(
                    labels=['Onshore', 'Offshore'],
                    values=[int(df_store['Onshore count']), int(df_store['Offshore count'])],
                    hole=.6,
                    marker=dict(colors=colors),
                    texttemplate='%{value} sites<br>(%{percent})',
                    hoverinfo='label+value+percent'
                )])
                site_distribution.update_layout(
                    title="Site Distribution (Count)",
                    showlegend=True,
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                st.plotly_chart(site_distribution, use_container_width=True)
                
                # 2. Bar Chart - Capacity Comparison
                capacity_comparison = go.Figure()
                capacity_comparison.add_trace(go.Bar(
                    x=['Onshore', 'Offshore'],
                    y=[onshore_capacity_mt, offshore_capacity_mt],
                    marker_color=colors,
                    text=[f"{onshore_capacity_mt:,.1f} Mt", f"{offshore_capacity_mt:,.1f} Mt"],
                    textposition='auto'
                ))
                capacity_comparison.update_layout(
                    title="Storage Capacity by Location",
                    xaxis_title="",
                    yaxis_title="Capacity (Mt)",
                    showlegend=False
                )
                st.plotly_chart(capacity_comparison, use_container_width=True)




        # --- NODES ---

            st.divider()
            st.markdown('###')
            st.subheader("Sites Map")
            
            nodes = nodes_map(df_source,
                            df_sink,
                            source_id='name',
                            source_lat='lat',
                            source_lon='lon',
                            sink_id='id',
                            sink_lat='latitude',
                            sink_lon='longitude')
            
            nodes.save(f"{os.getcwd()}/fe_func/nodes_map.html")
            nodes = open(f"{os.getcwd()}/fe_func/nodes_map.html")
            st.components.v1.html(nodes.read(), height=500, scrolling=True)

        else:
            st.write("No data available")