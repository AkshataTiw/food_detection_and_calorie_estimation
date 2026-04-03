import os
import warnings
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from skimage.measure import label, regionprops

warnings.filterwarnings("ignore")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Multi Class Food Detection and Calorie Estimation",
    page_icon="🍽️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.2rem;
}
.sub-text {
    text-align: center;
    font-size: 1.1rem;
    color: #b0b0b0;
    margin-bottom: 2rem;
}
.card {
    padding: 1.2rem;
    border-radius: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.metric-card {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(29,78,216,0.18), rgba(236,72,153,0.18));
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: white;
}
.metric-label {
    font-size: 1rem;
    color: #d1d5db;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-size: 1rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍽️ Multi Class Food Detection and Calorie Estimation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a food image to detect multiple food items, estimate their weight, and calculate total calories.</div>', unsafe_allow_html=True)

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = "best_new.pt"
CALIBRATION_PATH = "calibration.csv"
NUTRITION_PATH = "nutrition.csv"
COUNT_CONFIG_PATH = "count_based_config.csv"
MODELS_DIR = "models"

# =====================================================
# REQUIRED FILE CHECK
# =====================================================
required_files = [
    MODEL_PATH,
    CALIBRATION_PATH,
    NUTRITION_PATH,
    COUNT_CONFIG_PATH,
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if not os.path.isdir(MODELS_DIR):
    missing_files.append("models/")

if missing_files:
    st.error(f"Missing required files/folders: {', '.join(missing_files)}")
    st.stop()

# =====================================================
# LOAD RESOURCES
# =====================================================
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ YOLO model failed to load: {e}")
        st.stop()

@st.cache_data
def load_calibration():
    df = pd.read_csv(CALIBRATION_PATH)
    df["food"] = df["food"].astype(str).str.lower().str.strip()
    return df

@st.cache_data
def load_nutrition():
    df = pd.read_csv(NUTRITION_PATH)
    df["food"] = df["food"].astype(str).str.lower().str.strip()
    return df

@st.cache_data
def load_count_config():
    df = pd.read_csv(COUNT_CONFIG_PATH)
    df["food"] = df["food"].astype(str).str.lower().str.strip()
    return df

# 👉 Lazy load model (fix)
model_det = None

calib_df = load_calibration()
nutrition_df = load_nutrition()
count_df = load_count_config()

calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))
count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

# =====================================================
# HELPERS
# =====================================================
def extract_features_from_mask(mask):
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return None

    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    mask_area = np.sum(mask)
    height, width = mask.shape
    bbox_area = width * height

    labeled = label(mask)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return None

    region = max(regions, key=lambda r: r.area)

    perimeter = region.perimeter
    convex_area = getattr(region, "area_convex", None)
    if convex_area is None:
        convex_area = getattr(region, "convex_area", mask_area)

    major_axis = getattr(region, "axis_major_length", None)
    if major_axis is None:
        major_axis = getattr(region, "major_axis_length", 0.0)

    minor_axis = getattr(region, "axis_minor_length", None)
    if minor_axis is None:
        minor_axis = getattr(region, "minor_axis_length", 0.0)

    area_ratio = mask_area / (bbox_area + 1e-6)
    aspect_ratio = width / (height + 1e-6)
    solidity = mask_area / (convex_area + 1e-6)
    eccentricity = getattr(region, "eccentricity", 0.0)

    equiv_diameter = np.sqrt(4 * mask_area / np.pi)
    thickness = mask_area / (bbox_area + 1e-6)
    volume_proxy = (equiv_diameter ** 2) * thickness

    roundness = (4 * np.pi * mask_area) / (perimeter ** 2 + 1e-6)
    compactness = (perimeter ** 2) / (mask_area + 1e-6)

    elongation = major_axis / (minor_axis + 1e-6)
    fill_ratio = mask_area / (convex_area + 1e-6)

    return {
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "equiv_diameter": equiv_diameter,
        "thickness": thickness,
        "volume_proxy": volume_proxy,
        "roundness": roundness,
        "compactness": compactness,
        "elongation": elongation,
        "fill_ratio": fill_ratio,
    }

def predict_weight_regression(food, feature_dict):
    xgb_path = os.path.join(MODELS_DIR, f"xgb_{food}.pkl")
    rf_path = os.path.join(MODELS_DIR, f"rf_{food}.pkl")
    cols_path = os.path.join(MODELS_DIR, f"cols_{food}.pkl")

    if not (os.path.exists(xgb_path) and os.path.exists(rf_path) and os.path.exists(cols_path)):
        return None

    xgb = joblib.load(xgb_path)
    rf = joblib.load(rf_path)
    cols = joblib.load(cols_path)

    features = pd.DataFrame([feature_dict])

    for col in cols:
        if col not in features.columns:
            features[col] = 0

    features = features[cols]

    pred_xgb = np.exp(xgb.predict(features)[0]) - 1
    pred_rf = np.exp(rf.predict(features)[0]) - 1
    pred = 0.5 * pred_xgb + 0.5 * pred_rf

    row = calib_df[calib_df["food"] == food]
    if len(row) > 0:
        pred = row["a"].values[0] * pred + row["b"].values[0]

    return max(float(pred), 0.0)

def run_prediction(image_path):
    results = model_det(image_path, conf=0.25)

    rows = []
    total_calories = 0.0
    count_items = {}
    count = 1

    annotated_image = None
    if results and len(results) > 0:
        annotated_image = results[0].plot()

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(masks)):
            mask = (masks[i] > 0.5).astype(np.uint8)
            food = model_det.names[int(classes[i])].lower().strip()

            if food in count_weight_dict:
                count_items[food] = count_items.get(food, 0) + 1
                continue

            feature_dict = extract_features_from_mask(mask)
            if feature_dict is None:
                continue

            pred_weight = predict_weight_regression(food, feature_dict)
            if pred_weight is None:
                continue

            kcal_per_100g = calorie_dict.get(food, 0)
            kcal = (pred_weight / 100) * kcal_per_100g
            total_calories += kcal

            rows.append({
                "S.No": count,
                "Item": food.title(),
                "Weight (g)": round(pred_weight, 2),
                "Calories (kcal)": round(kcal, 2)
            })
            count += 1

    for food, cnt in count_items.items():
        weight_per_item = count_weight_dict[food]
        total_weight = cnt * weight_per_item
        kcal = (total_weight / 100) * calorie_dict.get(food, 0)
        total_calories += kcal

        rows.append({
            "S.No": count,
            "Item": f"{food.title()} x {cnt}",
            "Method": "Count-based",
            "Weight (g)": round(total_weight, 2),
            "Calories (kcal)": round(kcal, 2)
        })
        count += 1

    result_df = pd.DataFrame(rows)
    return result_df, round(total_calories, 2), annotated_image

# =====================================================
# MAIN UI
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Detection Preview</div>', unsafe_allow_html=True)
        preview_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Now"):

        if model_det is None:
            with st.spinner("Loading model..."):
                model_det = load_yolo_model()

        with st.spinner("Analyzing image and estimating calories..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                temp_path = tmp_file.name
                image.save(temp_path)

            try:
                result_df, total_calories, annotated_image = run_prediction(temp_path)

                with col2:
                    if annotated_image is not None:
                        preview_placeholder.image(annotated_image, channels="BGR", use_container_width=True)
                    else:
                        preview_placeholder.warning("No detections found.")

                st.markdown("<br>", unsafe_allow_html=True)

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(result_df)}</div>
                        <div class="metric-label">Detected Items</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_col2:
                    total_weight = result_df["Weight (g)"].sum() if not result_df.empty else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_weight:.2f} g</div>
                        <div class="metric-label">Estimated Total Weight</div>
                    </div>
                    """, unsafe_allow_html=True)

                with metric_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_calories:.2f} kcal</div>
                        <div class="metric-label">Total Calories</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

                if result_df.empty:
                    st.warning("No valid food items were detected.")
                else:
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="food_prediction_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)