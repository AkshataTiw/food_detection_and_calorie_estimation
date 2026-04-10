import os
import warnings
import tempfile

import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from PIL import Image
from ultralytics import YOLO
from skimage.measure import label, regionprops

warnings.filterwarnings("ignore")

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
    raise FileNotFoundError(f"Missing required files/folders: {', '.join(missing_files)}")

# =====================================================
# LOAD RESOURCES
# =====================================================
print("Loading model and resources...")

model_det = YOLO(MODEL_PATH)

calib_df = pd.read_csv(CALIBRATION_PATH)
calib_df["food"] = calib_df["food"].astype(str).str.lower().str.strip()

nutrition_df = pd.read_csv(NUTRITION_PATH)
nutrition_df["food"] = nutrition_df["food"].astype(str).str.lower().str.strip()

count_df = pd.read_csv(COUNT_CONFIG_PATH)
count_df["food"] = count_df["food"].astype(str).str.lower().str.strip()

calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))
count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

print("Resources loaded successfully.")

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
    serial_no = 1
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
                "S.No": serial_no,
                "Detected Food Item": food.title(),
                "Estimation Method": "Regression",
                "Estimated Weight (g)": round(pred_weight, 2),
                "Estimated Calories (kcal)": round(kcal, 2)
            })
            serial_no += 1

    for food, cnt in count_items.items():
        weight_per_item = count_weight_dict[food]
        total_weight = cnt * weight_per_item
        kcal = (total_weight / 100) * calorie_dict.get(food, 0)
        total_calories += kcal

        rows.append({
            "S.No": serial_no,
            "Detected Food Item": f"{food.title()} x {cnt}",
            "Estimation Method": "Count-Based",
            "Estimated Weight (g)": round(total_weight, 2),
            "Estimated Calories (kcal)": round(kcal, 2)
        })
        serial_no += 1

    result_df = pd.DataFrame(rows)
    return result_df, round(total_calories, 2), annotated_image


def create_pie_chart(result_df):
    fig, ax = plt.subplots(figsize=(7, 7))

    if result_df.empty or "Estimated Calories (kcal)" not in result_df.columns:
        ax.text(0.5, 0.5, "No prediction data available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    labels = result_df["Detected Food Item"].tolist()
    values = result_df["Estimated Calories (kcal)"].tolist()

    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Calorie Contribution by Detected Food Item", fontsize=15)
    return fig


def save_results_csv(result_df):
    csv_path = "food_prediction_results.csv"
    result_df.to_csv(csv_path, index=False)
    return csv_path


def predict_food(image):
    if image is None:
        empty_df = pd.DataFrame(columns=[
            "S.No",
            "Detected Food Item",
            "Estimation Method",
            "Estimated Weight (g)",
            "Estimated Calories (kcal)"
        ])
        return (
            gr.update(visible=False),
            None,
            "<div class='message-box'>Please upload a food image to continue.</div>",
            "<div class='stats-grid'></div>",
            empty_df,
            None,
            None
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            temp_path = tmp_file.name
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                pil_image = Image.fromarray(image).convert("RGB")
            pil_image.save(temp_path)

        result_df, total_calories, annotated_image = run_prediction(temp_path)

        if annotated_image is not None:
            annotated_image = annotated_image[:, :, ::-1]

        if result_df.empty:
            empty_df = pd.DataFrame(columns=[
                "S.No",
                "Detected Food Item",
                "Estimated Weight (g)",
                "Estimated Calories (kcal)"
            ])
            return (
                gr.update(visible=True),
                annotated_image,
                "<div class='message-box'>No valid food items were detected in the uploaded image.</div>",
                "<div class='stats-grid'></div>",
                empty_df,
                None,
                create_pie_chart(result_df)
            )

        total_weight = result_df["Estimated Weight (g)"].sum()
        detected_count = len(result_df)

        summary_html = """
        <div class='message-box success-box'>
            Analysis completed successfully. The detected food items, estimated weights, and calorie values are displayed below.
        </div>
        """

        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{detected_count}</div>
                <div class="stat-label">Detected Items</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_weight:.2f} g</div>
                <div class="stat-label">Estimated Total Weight</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_calories:.2f} kcal</div>
                <div class="stat-label">Estimated Total Calories</div>
            </div>
        </div>
        """

        csv_file = save_results_csv(result_df)
        pie_chart = create_pie_chart(result_df)

        return (
            gr.update(visible=True),
            annotated_image,
            summary_html,
            stats_html,
            result_df,
            csv_file,
            pie_chart
        )

    except Exception as e:
        empty_df = pd.DataFrame(columns=[
            "S.No",
            "Detected Food Item",
            "Estimation Method",
            "Estimated Weight (g)",
            "Estimated Calories (kcal)"
        ])
        return (
            gr.update(visible=True),
            None,
            f"<div class='message-box error-box'>Prediction failed: {str(e)}</div>",
            "<div class='stats-grid'></div>",
            empty_df,
            None,
            None
        )

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================
# BEAUTIFUL UI
# =====================================================
custom_css = """
:root {
    --bg: #06070b;
    --panel: rgba(17, 24, 39, 0.78);
    --panel-2: rgba(15, 23, 42, 0.92);
    --border: rgba(255,255,255,0.08);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent1: #7c3aed;
    --accent2: #ec4899;
    --accent3: #06b6d4;
}

body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(124,58,237,0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(236,72,153,0.12), transparent 28%),
        linear-gradient(180deg, #05060a 0%, #090b12 100%);
    color: var(--text);
    font-family: Inter, Arial, sans-serif;
}

.gradio-container {
    max-width: 1250px !important;
    margin: 0 auto !important;
    padding-top: 18px !important;
}

.hero-wrap {
    text-align: center;
    margin-bottom: 22px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04);
    color: #dbeafe;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 12px;
    background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 40%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 860px;
    margin: 0 auto;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.7;
}

.main-panel {
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(17,24,39,0.72), rgba(10,15,28,0.9));
    border-radius: 24px;
    padding: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    backdrop-filter: blur(18px);
}

.section-heading {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
}

.section-subtext {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}

.upload-note {
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 6px;
}

button.primary-btn {
    background: linear-gradient(90deg, #7c3aed, #ec4899, #06b6d4) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    min-height: 52px !important;
    box-shadow: 0 10px 30px rgba(124,58,237,0.28) !important;
}

button.primary-btn:hover {
    filter: brightness(1.05);
}

.results-shell {
    margin-top: 22px;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(8,12,22,0.92), rgba(14,19,31,0.95));
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.28);
}

.message-box {
    border-radius: 16px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    font-size: 14px;
    line-height: 1.6;
}

.success-box {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.22);
}

.error-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin: 18px 0 8px 0;
}

.stat-card {
    padding: 20px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(6,182,212,0.12));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}

.stat-label {
    font-size: 13px;
    color: #cbd5e1;
    letter-spacing: 0.2px;
}

.result-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #f8fafc;
}

footer {
    display: none !important;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 32px;
    }
    .stats-grid {
        grid-template-columns: 1fr;
    }
}
"""

with gr.Blocks(css=custom_css, title="Multi-Class Food Detection and Calorie Estimation", theme=gr.themes.Base()) as demo:
    gr.HTML("""
        <div class="hero-wrap">
            <div class="hero-badge">AI-Powered Food Analysis System</div>
            <div class="hero-title">A Hybrid Computer Vision Framework for Food Detection and Calorie Estimation Using Deep Learning and Regression Models.</div>
            <div class="hero-subtitle">
                Upload a food image to identify multiple food items, estimate their weight, and calculate the corresponding calorie contribution with a clear visual summary.
            </div>
        </div>
    """)

    with gr.Column(elem_classes="main-panel"):
        gr.HTML("""
            <div class="section-heading">Upload Image</div>
            <div class="section-subtext">
                Select a food image to begin analysis. The system will detect the visible food items and generate an estimated nutritional summary.
            </div>
        """)

        with gr.Row():
            input_image = gr.Image(
                type="pil",
                label="Food Image",
                height=360
            )

        gr.HTML("<div class='upload-note'>Supported formats: JPG, JPEG, PNG</div>")

        predict_btn = gr.Button("Analyze Image", elem_classes="primary-btn")

    # Hidden results section at startup
    with gr.Column(visible=False, elem_classes="results-shell") as results_section:
        gr.HTML("<div class='result-title'>Analysis Results</div>")

        with gr.Row():
            output_image = gr.Image(
                type="numpy",
                label="Detected Food Items",
                height=420
            )

        summary_output = gr.HTML()
        stats_output = gr.HTML()

        result_table = gr.Dataframe(
            label="Detailed Prediction Summary",
            interactive=False,
            wrap=True
        )

        with gr.Row():
            csv_output = gr.File(label="Download Prediction Results")

        pie_chart_output = gr.Plot(label="Calorie Distribution Chart")

    predict_btn.click(
        fn=predict_food,
        inputs=input_image,
        outputs=[
            results_section,
            output_image,
            summary_output,
            stats_output,
            result_table,
            csv_output,
            pie_chart_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
