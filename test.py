import streamlit as st

st.write("TEST")

import sys
st.write("Python path:", sys.path)

try:
    import yaml
    st.write("YAML module loaded from:", yaml.__file__)
except ImportError as e:
    st.write("YAML import failed:", e)

