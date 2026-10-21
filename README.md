# 🌿 Plant Disease Recognition System

An AI-powered agricultural diagnostic web application built with **Streamlit** and **TensorFlow / Keras** that instantly identifies crop foliar diseases and provides actionable treatment plans. The system classifies **38 distinct categories** spanning **14 major agricultural crops**.

---

## ⚠️ Important Dataset & Background Notice

> [!IMPORTANT]
> **Optimized for Plain / Grey Background Leaf Images**
> 
> The underlying deep learning model was trained on the **PlantVillage dataset** (~87,000 images). In this dataset, individual crop leaves were photographed against **uniform grey or black cardboard backgrounds** under controlled lighting.
> 
> * **For Best Diagnostic Accuracy:** Input images should feature an isolated leaf centered against a **plain, neutral background (grey, white, or dark surface)** with clear lighting.
> * **Real-World Photos:** When photographing leaves outdoors in gardens or fields, ensure the leaf fills **at least 85–90% of the camera frame** to minimize background interference from soil, fingers, or surrounding foliage.

---

## 🔄 System Architecture & Workflow

The diagnostic pipeline processes leaf images through an end-to-end convolutional neural network:

```
┌────────────────────────────────────────────────────────┐
│                   Leaf Image Input                     │
│    (Close-up leaf photo on plain / grey background)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Image Preprocessing                   │
│         Resize to 128 × 128 pixels & Normalize         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Deep Convolutional Neural Network          │
│       (Extracts lesion patterns, shapes & textures)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                Softmax Probability Layer               │
│         Computes confidence scores across 38 classes   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     User Dashboard                     │
│    • Primary Diagnosis & Health Badge                  │
│    • Top 3 Diagnostic Probability Distribution         │
│    • Actionable Treatment & Management Guide           │
└────────────────────────────────────────────────────────┘
```

### Architecture Highlights:
* **Feature Extractor:** Convolutional Neural Network with transfer learning optimized for foliar pattern recognition.
* **Input Dimension:** 128 × 128 × 3 (RGB).
* **Classification:** Softmax output across 38 categories with confidence estimation and Top-3 probability breakdown.

---

## 🚀 Key Features

* **Instant AI Diagnosis:** Real-time prediction classifying leaves across 38 crop disease categories.
* **Top-3 Probability Breakdown:** Diagnostic transparency displaying the top 3 candidate conditions with dynamic percentage bars.
* **Confidence Safeguards:** Alerts users when confidence is low (<50%) or moderate (<70%), suggesting better framing.
* **Management & Remedy Guide:** Symptoms checklist and actionable treatment recommendations for detected diseases.
* **Interactive UI:** Clean Streamlit dashboard with responsive layout and reference drawers.

---

## 🌱 Supported Crops & Disease Categories (38 Classes)

The system recognizes healthy and diseased conditions across **14 agricultural crops**:

| Crop | Recognizable Conditions |
| :--- | :--- |
| 🍎 **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 **Blueberry** | Healthy |
| 🍒 **Cherry** | Powdery Mildew, Healthy |
| 🌽 **Corn (Maize)** | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| 🍇 **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| 🍊 **Orange** | Huanglongbing (Citrus Greening) |
| 🍑 **Peach** | Bacterial Spot, Healthy |
| 🫑 **Pepper (Bell)** | Bacterial Spot, Healthy |
| 🥔 **Potato** | Early Blight, Late Blight, Healthy |
| 🍇 **Raspberry** | Healthy |
| 🌱 **Soybean** | Healthy |
| 🎃 **Squash** | Powdery Mildew |
| 🍓 **Strawberry** | Leaf Scorch, Healthy |
| 🍅 **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/raghavxd11/Plant-Disease.git
cd Plant-Disease
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard with:

```bash
python -m streamlit run main.py
```

Once started, open `http://localhost:8501` in your browser:
1. Navigate to the **Disease Recognition** page.
2. Upload a close-up photo of the leaf (on a plain/grey background for best accuracy).
3. Click **🚀 Analyze & Predict** to view the diagnosis, confidence breakdown, and remedies.

---

## 📁 Repository Structure

```text
Plant-Disease/
├── home_page.jpeg                     # Dashboard banner image
├── main.py                            # Streamlit web application & inference pipeline
├── README.md                          # Project documentation and guidelines
├── requirement.txt                    # Project dependencies
├── Test_plant_disease.ipynb           # Model testing & validation notebook
├── Train_plant_disease.ipynb          # Model architecture & training notebook
└── trained_plant_disease_model.keras  # Trained TensorFlow deep learning model
```