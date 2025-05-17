import streamlit as st


st.write("TEST")


import sys
print("Python path:", sys.path)

try:
    import yaml
    print("YAML module loaded from:", yaml.__file__)
except ImportError as e:
    print("YAML import failed:", e)