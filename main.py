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

# ----------------- DATA & HELPERS -----------------
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spo
# Class taxonomy defined
