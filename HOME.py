import streamlit as st
import plotly.graph_objects as go

import os



# --- PAGE CONFIG --- 
st.set_page_config(page_title="FoliaStream - CO₂ Network", layout="wide")

with st.sidebar:
    st.subheader("DATA SOURCES:")
    st.link_button("Climate TRACE", "https://climatetrace.org/", use_container_width=True)
    st.link_button("Oil & Gas Climate Initiative (OGCI)", "https://www.ogci.com/", use_container_width=True)


# --- LOGO-GLOBE ---
col1,col2,col3 = st.columns([1,2,1])
with col2:
    st.image(f"{os.getcwd()}/logo.png")


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