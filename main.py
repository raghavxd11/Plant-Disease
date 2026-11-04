import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
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
    "Huanglongbing (Citrus Greening)": {"symptoms": "Asymmetric blotchy yellowing on leaves.", "treatment": "Manage Asian citrus psyllid vectors; remove infected trees."},
    "Bacterial Spot": {"symptoms": "Small water-soaked angular spots turning dark.", "treatment": "Apply fixed copper mixed with mancozeb; avoid working in wet foliage."},
    "Early Blight": {"symptoms": "Dark brown concentric target-like rings on older leaves.", "treatment": "Apply chlorothalonil or copper soap; practice 3-year crop rotation."},
    "Late Blight": {"symptoms": "Rapidly expanding water-soaked lesions with white mold.", "treatment": "Destroy severely infected plants immediately; apply protectant fungicides."},
    "Leaf Mold": {"symptoms": "Pale green/yellow spots on top, olive mold underneath.", "treatment": "Improve ventilation in greenhouse; reduce humidity below 85%."},
    "Septoria Leaf Spot": {"symptoms": "Tiny circular spots with gray centers and black specks.", "treatment": "Mulch to prevent soil splash; remove lower infected leaves."},
    "Spider Mites": {"symptoms": "Fine yellow stippling and webbing on underside.", "treatment": "Spray insecticidal soap, horticultural oil, or introduce predatory mites."},
    "Target Spot": {"symptoms": "Brown circular lesions with faint concentric rings.", "treatment": "Ensure good plant spacing; spray azoxystrobin or copper fungicides."},
    "Tomato Yellow Leaf Curl Virus": {"symptoms": "Upward curling and severe yellowing of leaves.", "treatment": "Control whitefly vectors using yellow sticky traps and netting."},
    "Tomato Mosaic Virus": {"symptoms": "Mottled light and dark green mosaic patterns.", "treatment": "Sanitize tools with trisodium phosphate; destroy infected plants."},
    "Leaf Scorch": {"symptoms": "Purplish-brown spots expanding into scorched margins.", "treatment": "Prune old leaves after harvest; avoid excess nitrogen fertilizer."}
}

IMG_SIZE = (128, 128)

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

def format_class_name(raw_name: str):
    parts = raw_name.split("___")
    plant = parts[0].replace("_", " ").title().strip()
    disease = parts[1].replace("_", " ").title().strip() if len(parts) > 1 else "Unknown"
    # The dataset folder misspells "Huanglongbing"; fix it for display only.
    disease = disease.replace("Haunglongbing", "Huanglongbing")
    is_healthy = "healthy" in raw_name.lower()
    return plant, disease, is_healthy

def preprocess_for_model(image: Image.Image) -> np.ndarray:
    """Resize to 128x128 and wrap as a batch of 1.

    Pixel scaling (0-255 -> -1..1) is built into the saved model, so no /255 here.
    """
    resized = image.resize(IMG_SIZE)
    return np.array([np.asarray(resized, dtype=np.float32)])  # (1, 128, 128, 3)

MIN_LEAF_FRACTION = 0.02  # below this share of green pixels we assume "no leaf in photo"

def segment_leaf(image: Image.Image):
    """Find the leaf with classic DIP steps.

    1. Excess Green vegetation index, ExG = 2G - R - B (a point operation):
       high on leaves, ~0 on grey/white backgrounds, low on brown soil.
    2. Otsu threshold on the ExG image (with a floor, so a leaf must be clearly
       greener than grey).
    3. Morphological opening (remove specks) and closing (fill holes from lesions).
    4. Keep the largest connected component = the leaf.

    Returns (leaf_mask_image, box, area_fraction); box = (x, y, w, h) or None.
    """
    rgb = np.asarray(image)
    f = rgb.astype(np.float32)
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]                   # range -510..510
    exg8 = np.clip((exg + 255) / 2, 0, 255).astype(np.uint8)      # map to 0..255 (grey -> 127)

    otsu_t, _ = cv2.threshold(exg8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (exg8 >= max(otsu_t, 135)).astype(np.uint8) * 255      # 135 ~ ExG >= 15

    k = max(3, int(min(rgb.shape[:2]) * 0.01) | 1)  # odd kernel, ~1% of image size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return Image.fromarray(mask), None, 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area_frac = float(stats[largest, cv2.CC_STAT_AREA] / mask.size)
    leaf_mask = (labels == largest).astype(np.uint8) * 255
    box = tuple(int(v) for v in stats[largest, :4])
    return Image.fromarray(leaf_mask), box, area_frac

def auto_crop_leaf(image: Image.Image, scale: float = 1.4, max_fill: float = 0.3):
    """Crop a SQUARE around the segmented leaf, `scale` x its longer side, so the
    leaf sits in the frame like in the training photos and the later resize to
    128x128 does not stretch it.

    Photos where the leaf's box already covers more than `max_fill` of the image
    are left alone (cropping them only hurts accuracy).
    Returns (cropped_image, mask_image, cropped). If no leaf region is found or
    no crop is needed, returns the original image with cropped=False.
    """
    leaf_mask, box, area_frac = segment_leaf(image)
    if box is None or area_frac < MIN_LEAF_FRACTION:
        return image, leaf_mask, False

    x, y, w, h = box
    W, H = image.size
    if (w * h) / (W * H) > max_fill:  # leaf already fills the frame
        return image, leaf_mask, False

    side = int(max(w, h) * scale)
    side = min(side, W, H)
    cx, cy = x + w // 2, y + h // 2
    x0 = min(max(0, cx - side // 2), W - side)
    y0 = min(max(0, cy - side // 2), H - side)
    return image.crop((x0, y0, x0 + side, y0 + side)), leaf_mask, True

def get_remedy(disease_name: str, is_healthy: bool):
    if is_healthy:
        return {"symptoms": "Foliage appears vigorous and disease-free.", "treatment": "Maintain optimal watering and balanced fertilization."}
    for key, info in DISEASE_REMEDIES.items():
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return info
    return {"symptoms": "Foliar pathogen symptoms detected.", "treatment": "Isolate the plant, avoid wetting leaves, and consider a broad-spectrum organic copper fungicide."}

# ----------------- SIDEBAR -----------------
st.sidebar.title("🌿 Navigation")
app_mode = st.sidebar.radio("Go to:", ["Home", "Disease Recognition", "About"])

# ----------------- HOME PAGE -----------------
if app_mode == "Home":
    st.markdown('<div class="main-title">🌿 Plant Disease Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered diagnostic assistant for 38 crop disease categories across 14 plant species.</div>', unsafe_allow_html=True)

    if os.path.exists("home_page.jpeg"):
        st.image("home_page.jpeg", width=700)
    
    st.markdown("""
    ### How It Works
    1. **Upload Image:** Go to **Disease Recognition** and upload a close-up photo of a crop leaf.
    2. **AI Diagnosis:** The deep learning model analyzes visual patterns and returns the top matched diseases.
    3. **Treatment Guide:** View recommended symptoms checklist and actionable remedies.
    
    > **Tip for Best Accuracy:** The model is trained on close-up crop leaves. For real field photos, ensure the leaf fills the frame and minimize background soil/fingers.
    """)

# ----------------- DISEASE RECOGNITION PAGE -----------------
elif app_mode == "Disease Recognition":
    st.markdown('<div class="main-title">🔬 Disease Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a leaf photo to diagnose crop health and view actionable treatment recommendations.</div>', unsafe_allow_html=True)

    try:
        model = load_trained_model()
    except Exception as e:
        st.error(f"Could not load model file 'trained_plant_disease_model.keras': {e}")
        st.stop()

    test_image = st.file_uploader("Upload a Leaf Image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        image = Image.open(test_image).convert("RGB")
        use_crop = st.checkbox(
            "✂️ Auto-crop to the leaf (helps when the leaf is small in the photo; leaves close-ups unchanged)",
            value=True,
        )
        model_image = image
        _, _, leaf_fraction = segment_leaf(image)
        if use_crop:
            model_image, leaf_mask, cropped = auto_crop_leaf(image)
            if not cropped:
                st.caption("No crop applied: the leaf already fills the photo, or no single leaf could be isolated.")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 📷 Leaf Image")
            st.image(model_image, width=420)
            if use_crop:
                with st.expander("Show segmentation steps"):
                    st.image([image, leaf_mask], caption=["Original", "Leaf mask (ExG + Otsu + morphology)"], width=200)
            st.caption("ℹ️ For optimal results, ensure the infected leaf is centered with minimal background clutter.")

        with col2:
            st.markdown("### 🔍 Diagnosis")
            if st.button("🚀 Analyze & Predict", type="primary", width=420):
                with st.spinner("Analyzing foliar patterns..."):
                    input_arr = preprocess_for_model(model_image)
                    predictions = model.predict(input_arr, verbose=0)[0]
                    top_indices = np.argsort(predictions)[-3:][::-1]
                    
                    best_index = int(top_indices[0])
                    best_confidence = float(predictions[best_index]) * 100
                    best_raw_class = CLASS_NAMES[best_index]
                    plant, disease, is_healthy = format_class_name(best_raw_class)
                    remedy = get_remedy(disease, is_healthy)

                def show_top3():
                    for rank, idx in enumerate(top_indices, start=1):
                        cand_plant, cand_dis, cand_healthy = format_class_name(CLASS_NAMES[idx])
                        cand_prob = float(predictions[idx]) * 100
                        tag = "Healthy" if cand_healthy else cand_dis
                        st.write(f"**{rank}. {cand_plant}** — *{tag}* ({cand_prob:.1f}%)")
                        st.progress(min(1.0, cand_prob / 100.0))

                # Leaf-presence check: the model always names one of its 38 classes, even
                # for a photo with no leaf (random noise scores ~98%), so its confidence
                # alone cannot catch this. A green-pixel check on the photo can.
                if leaf_fraction < MIN_LEAF_FRACTION:
                    st.error(
                        "🚫 **No leaf detected.** Less than 2% of this photo looks like green leaf tissue. "
                        "The model always picks one of its 38 classes, even for non-leaf photos, "
                        "so no diagnosis is shown. Upload a close-up of a single leaf."
                    )
                    with st.expander("Show the model's raw guess anyway (unreliable)"):
                        show_top3()
                else:
                    # Primary Prediction Banner
                    if is_healthy:
                        st.markdown(f'<div class="status-healthy">✅ {plant} - Healthy ({best_confidence:.1f}%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-disease">⚠️ {plant} - {disease} ({best_confidence:.1f}%)</div>', unsafe_allow_html=True)

                    # Confidence Warning / Guidance
                    if best_confidence < 50.0:
                        st.warning(
                            "⚠️ **Low Confidence Prediction (< 50%)**: "
                            "The model is uncertain. This typically happens if the photo has background clutter (hands, soil, sky), "
                            "poor lighting, or belongs to an unsupported plant. Check the Top 3 candidates below or crop closer."
                        )
                    elif best_confidence < 70.0:
                        st.info("ℹ️ **Moderate Confidence**: Compare the Top 3 probable diagnoses below with visual symptoms.")

                    # Top 3 Candidates Breakdown
                    st.markdown("#### 📊 Top 3 Candidate Diagnoses")
                    show_top3()

                    # Remedy Card
                    # Each st.markdown call is rendered separately, so an opening <div> and a
                    # closing </div> in different calls never wrap anything. Build it in one call.
                    st.markdown(
                        f"""<div class="remedy-box">
<h4>📋 Recommended Management: {disease if not is_healthy else 'General Care'}</h4>
<p><b>Typical Symptoms:</b> {remedy['symptoms']}</p>
<p><b>Recommended Action:</b> {remedy['treatment']}</p>
</div>""",
                        unsafe_allow_html=True,
                    )

    with st.expander("🌱 View 14 Supported Crops & 38 Diagnosable Categories"):
        st.markdown("""
        The current model recognizes 38 specific classes across these 14 crops:
        - **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
        - **Blueberry**: Healthy
        - **Cherry**: Powdery Mildew, Healthy
        - **Corn (Maize)**: Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy
        - **Grape**: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy
        - **Orange**: Huanglongbing (Citrus Greening)
        - **Peach**: Bacterial Spot, Healthy
        - **Pepper (Bell)**: Bacterial Spot, Healthy
        - **Potato**: Early Blight, Late Blight, Healthy
        - **Raspberry**: Healthy
        - **Soybean**: Healthy
        - **Squash**: Powdery Mildew
        - **Strawberry**: Leaf Scorch, Healthy
        - **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
        """)

# ----------------- ABOUT PAGE -----------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is based on the PlantVillage dataset (~87,000 RGB images of healthy and diseased crop leaves across 38 classes).
    
    #### System Features
    - **Top-3 Multi-Class Output:** Provides diagnostic transparency with confidence distribution.
    - **No-Leaf Check:** An image-processing check (Excess Green index + Otsu threshold) blocks the diagnosis when the photo contains no leaf; low-confidence warnings flag uncertain predictions.
    - **Leaf Auto-Crop (optional):** Segments the leaf and crops to it, so the leaf fills the frame like in the training photos.
    - **Trained with Transfer Learning:** MobileNetV2 with real-world data augmentation.
    """)