import os
import warnings
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
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

st.title("🍽️ Food Detection & Calorie Estimation")

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = "best_new.pt"
CALIBRATION_PATH = "calibration.csv"
NUTRITION_PATH = "nutrition.csv"
COUNT_CONFIG_PATH = "count_based_config.csv"
MODELS_DIR = "models"

# =====================================================
# CHECK FILES
# =====================================================
required_files = [MODEL_PATH, CALIBRATION_PATH, NUTRITION_PATH, COUNT_CONFIG_PATH]
missing = [f for f in required_files if not os.path.exists(f)]

if not os.path.isdir(MODELS_DIR):
    missing.append("models/")

if missing:
    st.error(f"Missing: {', '.join(missing)}")
    st.stop()

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)

# =====================================================
# LOAD CSV
# =====================================================
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df["food"] = df["food"].str.lower().str.strip()
    return df

calib_df = load_csv(CALIBRATION_PATH)
nutrition_df = load_csv(NUTRITION_PATH)
count_df = load_csv(COUNT_CONFIG_PATH)

calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))
count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

# =====================================================
# SESSION MODEL
# =====================================================
if "model_det" not in st.session_state:
    st.session_state.model_det = None

# =====================================================
# FEATURE EXTRACTION
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
    if not regions:
        return None

    region = max(regions, key=lambda r: r.area)

    perimeter = region.perimeter
    convex_area = getattr(region, "convex_area", mask_area)

    major_axis = getattr(region, "axis_major_length", 0)
    minor_axis = getattr(region, "axis_minor_length", 1)

    return {
        "area_ratio": mask_area / (bbox_area + 1e-6),
        "aspect_ratio": width / (height + 1e-6),
        "solidity": mask_area / (convex_area + 1e-6),
        "eccentricity": getattr(region, "eccentricity", 0),
        "equiv_diameter": np.sqrt(4 * mask_area / np.pi),
        "thickness": mask_area / (bbox_area + 1e-6),
        "volume_proxy": mask_area,
        "roundness": (4 * np.pi * mask_area) / (perimeter**2 + 1e-6),
        "compactness": (perimeter**2) / (mask_area + 1e-6),
        "elongation": major_axis / (minor_axis + 1e-6),
        "fill_ratio": mask_area / (convex_area + 1e-6),
    }

# =====================================================
# SAFE WEIGHT PREDICTION
# =====================================================
def predict_weight_regression(food, feature_dict):
    try:
        xgb_path = os.path.join(MODELS_DIR, f"xgb_{food}.pkl")
        rf_path = os.path.join(MODELS_DIR, f"rf_{food}.pkl")
        cols_path = os.path.join(MODELS_DIR, f"cols_{food}.pkl")

        if not (os.path.exists(xgb_path) and os.path.exists(rf_path) and os.path.exists(cols_path)):
            return None

        xgb = joblib.load(xgb_path)
        rf = joblib.load(rf_path)
        cols = joblib.load(cols_path)

        df = pd.DataFrame([feature_dict])

        for col in cols:
            if col not in df:
                df[col] = 0

        df = df[cols]

        # ---------- SAFE XGBOOST ----------
        try:
            import xgboost as xgb_lib
            dmatrix = xgb_lib.DMatrix(df)
            xgb_pred = xgb.get_booster().predict(dmatrix)[0]
            xgb_pred = np.exp(xgb_pred) - 1
        except Exception as e:
            st.warning(f"XGB failed for {food}")
            xgb_pred = 0

        # ---------- RANDOM FOREST ----------
        try:
            rf_pred = np.exp(rf.predict(df)[0]) - 1
        except Exception as e:
            st.warning(f"RF failed for {food}")
            rf_pred = 0

        pred = 0.5 * (xgb_pred + rf_pred)

        # calibration
        row = calib_df[calib_df["food"] == food]
        if not row.empty:
            pred = row["a"].values[0] * pred + row["b"].values[0]

        return max(float(pred), 0)

    except Exception as e:
        st.error(f"Model failed for {food}: {e}")
        return None

# =====================================================
# MAIN PREDICTION
# =====================================================
def run_prediction(image_path):
    model = st.session_state.model_det
    results = model(image_path, conf=0.25)

    rows, total_calories = [], 0
    count_items = {}
    idx = 1

    annotated = results[0].plot() if results else None

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(masks)):
            mask = (masks[i] > 0.5).astype(np.uint8)
            food = model.names[int(classes[i])].lower().strip()

            if food in count_weight_dict:
                count_items[food] = count_items.get(food, 0) + 1
                continue

            feat = extract_features_from_mask(mask)
            if feat is None:
                continue

            weight = predict_weight_regression(food, feat)

            if weight is None:
                continue

            kcal = (weight / 100) * calorie_dict.get(food, 0)
            total_calories += kcal

            rows.append({
                "S.No": idx,
                "Item": food.title(),
                "Weight (g)": round(weight, 2),
                "Calories (kcal)": round(kcal, 2)
            })
            idx += 1

    # count-based foods
    for food, cnt in count_items.items():
        total_w = cnt * count_weight_dict[food]
        kcal = (total_w / 100) * calorie_dict.get(food, 0)
        total_calories += kcal

        rows.append({
            "S.No": idx,
            "Item": f"{food.title()} x {cnt}",
            "Weight (g)": round(total_w, 2),
            "Calories (kcal)": round(kcal, 2)
        })
        idx += 1

    return pd.DataFrame(rows), round(total_calories, 2), annotated

# =====================================================
# UI
# =====================================================
file = st.file_uploader("Upload image", type=["jpg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image)

    if st.button("🔍 Predict Now"):

        if st.session_state.model_det is None:
            st.session_state.model_det = load_yolo_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            path = tmp.name

        df, total_cal, img = run_prediction(path)

        if img is not None:
            st.image(img, channels="BGR")

        st.write("### Results")
        st.dataframe(df)

        st.success(f"🔥 Total Calories: {total_cal} kcal")

        os.remove(path)