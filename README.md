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
* **Feature Extractor:** MobileNetV2 pre-trained on ImageNet, kept frozen (transfer learning).
* **Built into the model:** data augmentation (flip, rotate, zoom, contrast; active only during training) and pixel rescaling to [−1, 1], so the app passes raw 0–255 pixels.
* **Classifier head:** Global Average Pooling → Dropout 0.3 → Dense 512 (ReLU) → Dropout 0.2 → Dense 38 (softmax).
* **Input Dimension:** 128 × 128 × 3 (RGB), resized with bilinear interpolation in both training and the app.
* **Classification:** Softmax output across 38 categories with confidence estimation and Top-3 probability breakdown.

---

## 🧪 Training & Evaluation

| Notebook | Model it trains | Status |
| :--- | :--- | :--- |
| `Train_mobilenetv2.ipynb` | **MobileNetV2 transfer learning: the model the app uses** | Current |
| `Train_plant_disease.ipynb` | Custom 10-layer CNN trained from scratch | Baseline only, not used by the app |
| `Test_plant_disease.ipynb` | Single-image prediction demo | — |

`Train_mobilenetv2.ipynb` splits `train/` 90/10 into train/validation, keeps `valid/` as an untouched **test set**, uses early stopping, and reports accuracy, per-class precision/recall and a confusion matrix.

> [!NOTE]
> The 95.75% accuracy in `Train_plant_disease.ipynb` belongs to the **baseline CNN**, not the MobileNetV2 model used by the app.

---

## 🚀 Key Features

* **Instant AI Diagnosis:** Real-time prediction classifying leaves across 38 crop disease categories.
* **Top-3 Probability Breakdown:** Diagnostic transparency displaying the top 3 candidate conditions with dynamic percentage bars.
* **Confidence Safeguards:** Alerts users when confidence is low (<50%) or moderate (<70%), suggesting better framing.
* **Management & Remedy Guide:** Symptoms checklist and actionable treatment recommendations for detected diseases.
* **Interactive UI:** Clean Streamlit dashboard with responsive layout and reference drawers.
* **Leaf Auto-Crop & No-Leaf Check:** Classic image-processing step (Excess Green index ExG = 2G − R − B → Otsu threshold → morphological opening/closing → largest connected component) that crops photos where the leaf is small, and blocks the diagnosis when no leaf is present. The segmentation mask is shown for inspection.

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
3. **✂️ Auto-crop to the leaf** is on by default; it only crops when the leaf is small in the photo.
4. Click **🚀 Analyze & Predict** to view the diagnosis, confidence breakdown, and remedies.

---

## ⚠️ Known Limitations

* **Lab-style training data:** PlantVillage leaves are photographed on plain backgrounds, so accuracy drops on real field photos. Auto-crop helps but does not remove this gap; retraining with field images would.
* **Closed set of 38 classes:** the model always picks one of its 38 classes (random noise scores ~98% confidence). The no-leaf check catches photos with no green leaf, but an *unsupported* plant's leaf (e.g. a rose) will still be given one of the 38 labels.
* **Optimistic test score:** the Kaggle dataset was built with offline augmentation, so near-duplicates of a leaf can appear in both `train/` and `valid/`.
* **Remedies are a fixed lookup table** written by hand, not generated or verified by the model. Consult a local agricultural expert before treating crops.

---

## 📁 Repository Structure

```text
Plant-Disease/
├── home_page.jpeg                     # Dashboard banner image
├── main.py                            # Streamlit web application & inference pipeline
├── README.md                          # Project documentation and guidelines
├── requirement.txt                    # Project dependencies
├── Test_plant_disease.ipynb           # Single-image prediction demo
├── Train_mobilenetv2.ipynb            # Trains + evaluates the MobileNetV2 model the app uses
├── Train_plant_disease.ipynb          # Baseline custom CNN (not used by the app)
└── trained_plant_disease_model.keras  # Shipped MobileNetV2 model loaded by main.py
```