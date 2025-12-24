import streamlit as st
import plotly.graph_objects as go

import os

from streamlit_autorefresh import st_autorefresh


# --- PAGE CONFIG --- 
st.set_page_config(page_title="FoliaStream - CO₂ Network", layout="wide")

with st.sidebar:
    st.subheader("DATA SOURCES:")
    st.link_button("Climate TRACE", "https://climatetrace.org/", use_container_width=True)
    st.link_button("Oil & Gas Climate Initiative (OGCI)", "https://www.ogci.com/", use_container_width=True)


# --- LOGO-GLOBE ---
col1, col2 = st.columns([2,1])
with col1:
    st.image(f"{os.getcwd()}/logo.png")



with col2:

    BORDER_COLOR = "rgb(0, 80, 200)"          
    BACKGROUND_COLOR = "rgb(245, 248, 250)"   
    OCEAN_COLOR = "rgb(220, 235, 255)"  
    COUNTRY_COLOR = "rgba(0, 80, 200, 0.4)"

    # ---- INIT STATE ----
    if "lon" not in st.session_state:
        st.session_state.lon = 0

    # ---- AUTO REFRESH (non-blocking) ----
    st_autorefresh(interval=150, key="globe_refresh")

    # ---- UPDATE ROTATION ----
    st.session_state.lon = (st.session_state.lon + 0.5) % 360

    # ---- GLOBE ----
    globe = go.Figure()
    globe.add_trace(go.Scattergeo(
        lon=[0],
        lat=[0],
        mode="markers",
        marker=dict(size=1, color="rgba(0,0,0,0)")
    ))

    globe.update_layout(
        geo=dict(
            projection=dict(
                type="orthographic",
                rotation=dict(lon=st.session_state.lon, lat=21)
            ),
            showland=True,
            landcolor='rgb(245, 248, 250)',
            showocean=True,
            oceancolor='rgb(220, 235, 255)',
            showcountries=True,
            countrycolor='rgba(0, 80, 200, 0.4)',
            countrywidth = 1.5,
            showcoastlines=True,
            coastlinecolor='rgb(0, 80, 200)',
            ),
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=0, r=0, t=0, b=0),
    # width=800,
    dragmode='orbit',
    autosize=True
    )

    st.plotly_chart(globe, use_container_width=True, config={"displayModeBar":False})

st.divider()

# --- DESCRIPTION ---
col1, col2 = st.columns([1,1])
with col1:
    st.header("Welcome to FoliaStream!")
    st.markdown("""
    This interactive tool helps you analyze and visualize Carbon Capture & Storage networks.
    """)

with col2:
    st.header("📌 Pages")
    st.page_link("pages/COUNTRY_OVERVIEW.py", label="📊 Country Overview")
    st.page_link("pages/NETWORK_MAP.py", label="🌍 Network Map")

st.divider()

# --- FOOTER ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("CONTACT")
with col2:
    st.markdown("📩 foliastream@gmail.com")
with col3:
    st.caption("🌐 https://foliastream.streamlit.app")
with col4:
    st.caption("</> https://github.com/FoliaStream/FoliaStream")