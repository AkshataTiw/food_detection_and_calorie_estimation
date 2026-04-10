import os
import warnings
import tempfile
import random

import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from skimage.measure import label, regionprops

warnings.filterwarnings("ignore")

# =====================================================
# DETERMINISM
# =====================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================
# MODEL CACHE
# =====================================================
model_cache = {}

def load_model(path):
    if path not in model_cache:
        model_cache[path] = joblib.load(path)
    return model_cache[path]

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = "best_new.pt"
CALIBRATION_PATH = "calibration.csv"
NUTRITION_PATH = "nutrition.csv"
COUNT_CONFIG_PATH = "count_based_config.csv"
MODELS_DIR = "models"

# =====================================================
# LOAD
# =====================================================
model_det = YOLO(MODEL_PATH)

calib_df = pd.read_csv(CALIBRATION_PATH)
calib_df["food"] = calib_df["food"].str.lower().str.strip()

nutrition_df = pd.read_csv(NUTRITION_PATH)
nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()

count_df = pd.read_csv(COUNT_CONFIG_PATH)
count_df["food"] = count_df["food"].str.lower().str.strip()

calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))
count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

# =====================================================
# PIE CHART
# =====================================================
def create_pie_chart(df):
    if df.empty:
        return None
    fig, ax = plt.subplots()
    ax.pie(
        df["Estimated Calories (kcal)"],
        labels=df["Detected Food Item"],
        autopct='%1.1f%%'
    )
    ax.set_title("Calorie Distribution")
    return fig

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features_from_mask(mask):
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return None

    x_min, x_max = x_idx.min(), x_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    mask = mask[y_min:y_max+1, x_min:x_max+1]

    mask_area = np.sum(mask)
    height, width = mask.shape
    bbox_area = width * height

    labeled = label(mask)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return None

    region = max(regions, key=lambda r: r.area)

    perimeter = region.perimeter
    convex_area = region.convex_area
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length

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
        "fill_ratio": fill_ratio
    }

# =====================================================
# PREDICT WEIGHT
# =====================================================
def predict_weight(food, feat):
    xgb = load_model(os.path.join(MODELS_DIR, f"xgb_{food}.pkl"))
    rf = load_model(os.path.join(MODELS_DIR, f"rf_{food}.pkl"))
    cols = load_model(os.path.join(MODELS_DIR, f"cols_{food}.pkl"))

    df = pd.DataFrame([feat])

    for c in cols:
        if c not in df.columns:
            df[c] = 0

    df = df[cols]

    pred = 0.5*(np.exp(xgb.predict(df)[0]) - 1) + \
           0.5*(np.exp(rf.predict(df)[0]) - 1)

    row = calib_df[calib_df["food"] == food]
    if not row.empty:
        pred = row["a"].values[0] * pred + row["b"].values[0]

    return max(pred, 0)

# =====================================================
# RUN PREDICTION
# =====================================================
def run_prediction(path):
    results = model_det(path, conf=0.25)

    rows = []
    total = 0
    sn = 1
    count_items = {}

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

            feat = extract_features_from_mask(mask)
            if feat is None:
                continue

            w = predict_weight(food, feat)
            kcal = (w / 100) * calorie_dict.get(food, 0)

            total += kcal

            rows.append({
                "S.No": sn,
                "Detected Food Item": food.title(),
                "Estimated Weight (g)": round(w, 2),
                "Estimated Calories (kcal)": round(kcal, 2)
            })
            sn += 1

    for food, cnt in count_items.items():
        w = cnt * count_weight_dict[food]
        kcal = (w / 100) * calorie_dict.get(food, 0)

        total += kcal

        rows.append({
            "S.No": sn,
            "Detected Food Item": f"{food.title()} x {cnt}",
            "Estimated Weight (g)": round(w, 2),
            "Estimated Calories (kcal)": round(kcal, 2)
        })
        sn += 1

    return pd.DataFrame(rows), round(total, 2), annotated

# =====================================================
# CSV SAVE
# =====================================================
def save_csv(df):
    path = os.path.join(tempfile.gettempdir(), "download_csv.csv")
    df.to_csv(path, index=False)
    return path

# =====================================================
# MAIN FUNCTION
# =====================================================
def predict_food(image):
    if image is None:
        return None, None, pd.DataFrame(), "Upload image", None, None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        path = tmp.name
        image.save(path)

    df, total, img = run_prediction(path)

    if img is not None:
        img = img[:, :, ::-1]

    pie = create_pie_chart(df)
    csv = save_csv(df)

    os.remove(path)

    return image, img, df, f"<h2>Total Calories: {total} kcal</h2>", csv, pie

# =====================================================
# UI
# =====================================================
with gr.Blocks(css="""
.container {max-width: 1000px; margin: auto; text-align: center;}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 5px;
}
.badge {
    display: inline-block;
    background: #111;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    margin-bottom: 20px;
}
.img-box img {
    height: 280px !important;
    object-fit: contain;
}
button {
    width: 200px;
    margin-top: 10px;
}
""") as demo:

    gr.Markdown(
        "<div class='title'>Food Detection and Calorie Estimation Using Deep Learning and Regression Models</div>"
    )

    gr.Markdown(
        "<div class='badge'>Powered by YOLO Detection</div>"
    )

    with gr.Column(elem_classes="container"):

        img = gr.Image(type="pil", label="Upload Food Image")
        btn = gr.Button("Analyze")

        with gr.Row():
            input_display = gr.Image(label="Uploaded Image", elem_classes="img-box")
            out_img = gr.Image(label="Detected Image", elem_classes="img-box")

        table = gr.Dataframe()

        total_html = gr.HTML()

        file = gr.File(label="Download CSV")

        pie_plot = gr.Plot()

    btn.click(
        predict_food,
        img,
        [input_display, out_img, table, total_html, file, pie_plot]
    )

demo.launch()