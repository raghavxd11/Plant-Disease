import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4b6b52;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-healthy {
        background-color: #e8f5e9;
        color: #1b5e20;
        border-left: 5px solid #2e7d32;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.15rem;
        margin: 10px 0;
    }
    .status-disease {
        background-color: #ffebee;
        color: #b71c1c;
        border-left: 5px solid #d32f2f;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.15rem;
        margin: 10px 0;
    }
    .remedy-box {
        background-color: #f9fbe7;
        border: 1px solid #dce775;
        border-radius: 8px;
        padding: 16px;
        margin-top: 15px;
    }
    .top-candidate {
        background-color: #f1f8e9;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------
# UI layout in progress
