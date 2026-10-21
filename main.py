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
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

DISEASE_REMEDIES = {
    "Apple Scab": {"symptoms": "Olive-green to black velvety spots on leaves.", "treatment": "Apply protectant fungicides (captan/mancozeb); prune canopy for airflow."},
    "Black Rot": {"symptoms": "Circular brown leaf lesions and rotting fruit.", "treatment": "Prune out dead wood; remove mummified fruit; apply copper fungicide."},
    "Cedar Apple Rust": {"symptoms": "Bright yellow-orange spots on leaves.", "treatment": "Apply myclobutanil or sulfur; eliminate nearby red cedar hosts if possible."},
    "Powdery Mildew": {"symptoms": "White powdery fungal coating on foliage.", "treatment": "Spray with wettable sulfur, potassium bicarbonate, or neem oil."},
    "Common Rust": {"symptoms": "Reddish-brown powdery pustules on leaf surfaces.", "treatment": "Plant rust-resistant hybrids; apply strobilurin fungicides if severe."},
    "Northern Leaf Blight": {"symptoms": "Long, elliptical grayish-green or tan lesions.", "treatment": "Rotate crops; till crop residue; apply foliar fungicides."},
    "Cercospora Leaf Spot": {"symptoms": "Small tan lesions with dark borders.", "treatment": "Improve air circulation; apply copper-based fungicides."},
    "Esca (Black Measles)": {"symptoms": "Tiger-stripe interveinal leaf discoloration.", "treatment": "Prune infected vines during dry periods; paint pruning wounds."},
    "Leaf Blight": {"symptoms": "Irregular necrotic brown lesions on foliage.", "treatment": "Apply copper sprays; avoid overhead irrigation."},
    "Haunglongbing (Citrus Greening)": {"symptoms": "Asymmetric blotchy yellowing on leaves.", "treatment": "Manage Asian citrus psyllid vectors; remove infected trees."},
    "Bacterial Spot": {"symptoms": "Small water-soaked angular spots turning dark.", "treatment": "Apply fixed copper mixed with mancozeb; avoid working in wet foliage."},
    "Early Blight": {"symptoms": "Dark brown concentric target-like rings on older leaves.", "treatment": "Apply chlorothalonil or copper soap; practice 3-year crop rotation."},
    "Late Blight": {"symptoms": "Rapidly expanding water-soaked lesions with white mold.", "treatment": "Destroy severely infected plants immediately; apply protectant fungicides."},
    "Leaf Mold": {"symptoms": "Pale green/yellow spots on top, olive mold underneath.", "treatment": "Improve ventilation in greenhouse; reduce humidity below 85%."},
    "Septoria Leaf Spot": {"symptoms": "Tiny circular spots with gray centers and black specks.", "treatment": "Mulch to prevent soil splash; remove lower infected leaves."},
    "Spider Mites": {"symptoms": "Fine yellow stippling and webbing on underside.", "treatment": "Spray insecticidal soap, horticultural oil, or introduce predatory mites."},
    "Target Spot": {"symptoms": "Brown circular lesions with faint concentric rin
# Cached model loader integrated
