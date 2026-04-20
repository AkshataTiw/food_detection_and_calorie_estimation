import os
import warnings
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ultralytics import YOLO
from skimage.measure import label, regionprops
from PIL import Image

warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Food AI", layout="wide")

# =========================
# CUSTOM CSS (PREMIUM UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.block-container {
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    text-align: center;
}
.metric {
    font-size: 28px;
    font-weight: bold;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🍽️ Food Detection and Calorie Estimation</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>YOLO + Feature Engineering + ML</div>", unsafe_allow_html=True)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_all():
    model = YOLO("best_new.pt")

    calib = pd.read_csv("calibration.csv")

    nutrition = pd.read_csv("nutrition.csv")
    nutrition["food"] = nutrition["food"].str.lower().str.strip()
    calorie_dict = dict(zip(nutrition["food"], nutrition["kcal_per_100g"]))

    count_df = pd.read_csv("count_based_config.csv")
    count_df["food"] = count_df["food"].str.lower().str.strip()
    count_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

    return model, calib, calorie_dict, count_dict

model_det, calib_df, calorie_dict, count_weight_dict = load_all()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(mask):
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return None

    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    mask = mask[y_min:y_max+1, x_min:x_max+1]

    mask_area = np.sum(mask)
    h, w = mask.shape
    bbox_area = w * h

    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return None

    r = max(regions, key=lambda x: x.area)

    perimeter = r.perimeter
    convex_area = r.convex_area

    return {
        "area_ratio": mask_area / (bbox_area + 1e-6),
        "aspect_ratio": w / (h + 1e-6),
        "solidity": mask_area / (convex_area + 1e-6),
        "eccentricity": r.eccentricity,
        "equiv_diameter": np.sqrt(4 * mask_area / np.pi),
        "thickness": mask_area / (bbox_area + 1e-6),
        "volume_proxy": mask_area,
        "roundness": (4 * np.pi * mask_area) / (perimeter**2 + 1e-6),
        "compactness": (perimeter**2) / (mask_area + 1e-6),
        "elongation": r.major_axis_length / (r.minor_axis_length + 1e-6),
        "fill_ratio": mask_area / (convex_area + 1e-6)
    }

# =========================
# PREDICTION
# =========================
def predict(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        path = tmp.name

    results = model_det(path, conf=0.25)

    rows = []
    total = 0
    count_items = {}
    sn = 1

    annotated = results[0].plot()

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(masks)):
            food = model_det.names[int(classes[i])].lower().strip()

            if food in count_weight_dict:
                count_items[food] = count_items.get(food, 0) + 1
                continue

            mask = (masks[i] > 0.5).astype(np.uint8)
            feat = extract_features(mask)
            if feat is None:
                continue

            xgb = joblib.load(f"models/xgb_{food}.pkl")
            rf = joblib.load(f"models/rf_{food}.pkl")
            cols = joblib.load(f"models/cols_{food}.pkl")

            df_feat = pd.DataFrame([feat])
            for c in cols:
                if c not in df_feat.columns:
                    df_feat[c] = 0
            df_feat = df_feat[cols]

            pred = 0.5*(np.exp(xgb.predict(df_feat)[0]) - 1) + \
                   0.5*(np.exp(rf.predict(df_feat)[0]) - 1)

            row = calib_df[calib_df["food"] == food]
            if not row.empty:
                pred = row["a"].values[0]*pred + row["b"].values[0]

            kcal = (pred/100) * calorie_dict.get(food, 0)
            total += kcal

            rows.append([sn, food.title(), round(pred,2), round(kcal,2)])
            sn += 1

    for food, cnt in count_items.items():
        w = cnt * count_weight_dict[food]
        kcal = (w/100) * calorie_dict.get(food,0)
        total += kcal
        rows.append([sn, f"{food} x {cnt}", w, round(kcal,2)])
        sn += 1

    os.remove(path)

    df = pd.DataFrame(rows, columns=["S.No","Food","Weight (g)","Calories"])
    return df, total, annotated

# =========================
# UI
# =========================
uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded")

    if st.button("🚀 Analyze"):
        with st.spinner("Analyzing..."):

            df, total, annotated = predict(image)

            with col2:
                st.image(annotated, caption="Detected")

            st.markdown("<br>", unsafe_allow_html=True)

            # TABLE
            st.markdown("### 📊 Detailed Results")
            st.dataframe(df, use_container_width=True)

            # 🔥 TOTAL BELOW TABLE
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='card'><div class='metric'>🔥 Total Calories: {round(total,2)} kcal</div></div>",
                unsafe_allow_html=True
            )

            # PIE CHART
            if not df.empty:
                fig, ax = plt.subplots()
                ax.pie(df["Calories"], labels=df["Food"], autopct="%1.1f%%")
                ax.set_title("Calorie Split")
                st.pyplot(fig)

            # DOWNLOAD CSV
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")