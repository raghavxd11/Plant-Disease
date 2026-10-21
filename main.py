import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Plant Disease Recognition", layout="wide")
st.sidebar.title("🌿 Navigation")
app_mode = st.sidebar.radio("Go to:", ["Home", "Disease Recognition", "About"])
