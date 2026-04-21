import os
import warnings
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from PIL import Image

warnings.filterwarnings("ignore")

# =========================
# SAFE YOLO IMPORT
# =========================
try:
    from ultralytics import YOLO
except Exception as e:
    st.error(f"❌ YOLO failed: {e}")
    st.stop()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NutriLens 🍽️", layout="wide")
st.markdown("<h1 style='text-align:center;'>🍽️ NutriLens</h1>", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_all():
    model = YOLO("best_new.pt")

    calib_df = pd.read_csv("calibration.csv")

    nutrition_df = pd.read_csv("nutrition.csv")
    nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()
    calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

    count_df = pd.read_csv("count_based_config.csv")
    count_df["food"] = count_df["food"].str.lower().str.strip()
    count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

    return model, calib_df, calorie_dict, count_weight_dict

model_det, calib_df, calorie_dict, count_weight_dict = load_all()

# =========================
# PREDICTION
# =========================
def predict(image):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    results = model_det(img_path, conf=0.25)

    total_calories = 0
    rows = []
    count = 1
    count_items = {}

    annotated = results[0].plot()

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(masks)):

            mask = (masks[i] > 0.5).astype(np.uint8)

            y_idx, x_idx = np.where(mask)
            if len(x_idx) == 0:
                continue

            x_min, x_max = x_idx.min(), x_idx.max()
            y_min, y_max = y_idx.min(), y_idx.max()
            mask = mask[y_min:y_max+1, x_min:x_max+1]

            mask_area = np.sum(mask)
            height, width = mask.shape
            bbox_area = width * height

            labeled = label(mask)
            regions = regionprops(labeled)
            if len(regions) == 0:
                continue

            region = max(regions, key=lambda r: r.area)

            perimeter = region.perimeter
            convex_area = region.area_convex
            major_axis = region.axis_major_length
            minor_axis = region.axis_minor_length

            food = model_det.names[int(classes[i])].lower().strip()

            # COUNT BASED
            if food in count_weight_dict:
                count_items[food] = count_items.get(food, 0) + 1
                continue

            # =========================
            # 🔥 FIXED MODEL LOADING
            # =========================
            base_path = os.path.join(os.getcwd(), "models")

            xgb_path = os.path.join(base_path, f"xgb_{food}.pkl")
            rf_path = os.path.join(base_path, f"rf_{food}.pkl")
            cols_path = os.path.join(base_path, f"cols_{food}.pkl")

            if not os.path.exists(xgb_path):
                st.warning(f"⚠️ Model missing for: {food}")
                continue

            try:
                xgb = joblib.load(xgb_path)
                rf = joblib.load(rf_path)
                cols = joblib.load(cols_path)
            except Exception as e:
                st.error(f"❌ Failed loading {food}: {e}")
                continue

            # FEATURES
            area_ratio = mask_area / (bbox_area + 1e-6)
            aspect_ratio = width / (height + 1e-6)
            solidity = mask_area / (convex_area + 1e-6)
            eccentricity = region.eccentricity

            equiv_diameter = np.sqrt(4 * mask_area / np.pi)
            thickness = mask_area / (bbox_area + 1e-6)
            volume_proxy = (equiv_diameter ** 2) * thickness

            roundness = (4 * np.pi * mask_area) / (perimeter**2 + 1e-6)
            compactness = (perimeter**2) / (mask_area + 1e-6)

            elongation = major_axis / (minor_axis + 1e-6)
            fill_ratio = mask_area / (convex_area + 1e-6)

            features = pd.DataFrame([{
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
                "fill_ratio": fill_ratio
            }])[cols]

            pred_xgb = np.exp(xgb.predict(features)[0]) - 1
            pred_rf = np.exp(rf.predict(features)[0]) - 1
            pred = 0.5 * pred_xgb + 0.5 * pred_rf

            row = calib_df[calib_df["food"] == food]
            if len(row) > 0:
                pred = row["a"].values[0] * pred + row["b"].values[0]

            kcal = (pred / 100) * calorie_dict.get(food, 0)
            total_calories += kcal

            rows.append([count, food.title(), round(pred,2), round(kcal,2)])
            count += 1

    # COUNT ITEMS
    for food, cnt in count_items.items():
        weight_per_item = count_weight_dict[food]
        total_weight = cnt * weight_per_item
        kcal = (total_weight / 100) * calorie_dict.get(food, 0)

        total_calories += kcal
        rows.append([count, f"{food} x {cnt}", round(total_weight,2), round(kcal,2)])
        count += 1

    os.remove(img_path)

    df = pd.DataFrame(rows, columns=["S.No","Food","Weight (g)","Calories"])
    return df, total_calories, annotated

# =========================
# UI
# =========================
uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=500)

    if st.button("🚀 Analyze"):
        with st.spinner("Analyzing..."):

            df, total, annotated = predict(image)

            with col2:
                st.image(annotated, caption="Detected", width=500)

            st.markdown("### 📊 Results")
            st.dataframe(df)

            st.markdown(f"## 🔥 Total Calories: {round(total,2)} kcal")

            if not df.empty and df["Calories"].sum() > 0:
                fig, ax = plt.subplots()
                ax.pie(df["Calories"], labels=df["Food"], autopct="%1.1f%%")
                st.pyplot(fig)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")