import os
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops
import warnings

warnings.filterwarnings("ignore")

# =========================
# LOAD MODELS
# =========================
model_det = YOLO("best_new.pt")

nutrition_df = pd.read_csv("nutrition.csv")
nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

img_path = "test_images/test27.jpeg"

results = model_det(img_path, conf=0.25, verbose=False)

total_calories = 0
results_list = []

# =========================
# PROCESS DETECTIONS
# =========================
for r in results:
    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    num_objects = len(masks)  # for multi-object correction

    for i, m in enumerate(masks):

        mask = (m > 0.5).astype(np.uint8)

        y_idx, x_idx = np.where(mask)
        if len(x_idx) == 0:
            continue

        class_id = int(classes[i])
        class_name = model_det.names[class_id].lower().strip()

        model_path = f"models/model_{class_name}.pkl"
        cols_path = f"models/columns_{class_name}.pkl"

        if not os.path.exists(model_path):
            print(f"⚠ No model for {class_name}")
            continue

        model_reg = joblib.load(model_path)
        model_columns = joblib.load(cols_path)

        # =========================
        # CROP MASK
        # =========================
        x_min, x_max = x_idx.min(), x_idx.max()
        y_min, y_max = y_idx.min(), y_idx.max()
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]

        mask_area = np.sum(mask_crop)

        labeled = label(mask_crop)
        regions = regionprops(labeled)

        if len(regions) == 0:
            continue

        region = max(regions, key=lambda r: r.area)

        perimeter = region.perimeter
        convex_area = region.area_convex

        width = mask_crop.shape[1]
        height = mask_crop.shape[0]

        # =========================
        # FEATURE EXTRACTION (MATCH TRAINING)
        # =========================
        area = mask_area
        aspect_ratio = width / (height + 1e-6)

        extent = area / (width * height + 1e-6)
        solidity = area / (convex_area + 1e-6)
        eccentricity = region.eccentricity

        equiv_diameter = np.sqrt(4 * area / np.pi)
        volume_proxy = equiv_diameter ** 3

        thickness_proxy = area / (width * height + 1e-6)

        roundness = (4 * np.pi * area) / (perimeter**2 + 1e-6)

        area_x_solidity = area * solidity
        area_x_thickness = area * thickness_proxy

        # =========================
        # CREATE FEATURE DF
        # =========================
        features = pd.DataFrame([{
            "area": area,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "equiv_diameter": equiv_diameter,
            "volume_proxy": volume_proxy,
            "thickness_proxy": thickness_proxy,
            "roundness": roundness,
            "area_x_solidity": area_x_solidity,
            "area_x_thickness": area_x_thickness
        }])

        features = features[model_columns]

        # =========================
        # PREDICT (LOG → NORMAL)
        # =========================
        pred_log = model_reg.predict(features)[0]
        pred_weight = np.exp(pred_log) - 1

        # =========================
        # MULTI-OBJECT CORRECTION
        # =========================
        if num_objects > 1:
            pred_weight *= (1 + 0.08 * (num_objects - 1))

        # =========================
        # STABILITY CLAMP
        # =========================
        pred_weight = np.clip(pred_weight, 10, 500)

        # =========================
        # CALORIE CALCULATION
        # =========================
        kcal = 0
        if class_name in calorie_dict:
            kcal = (pred_weight / 100) * calorie_dict[class_name]
            total_calories += kcal

        results_list.append({
            "item": class_name,
            "weight": round(pred_weight, 2),
            "calories": round(kcal, 2)
        })

# =========================
# SORT (OPTIONAL CLEANNESS)
# =========================
results_list = sorted(results_list, key=lambda x: x["item"])

# =========================
# CLEAN OUTPUT (IMPROVED FORMAT ONLY)
# =========================
print("\n🍽️ Food Prediction Summary")
print("-" * 50)

# Header
print(f"{'Item':<12} {'Weight(g)':>12} {'Calories (kcal)':>18}")
print("-" * 50)

# Rows
for r in results_list:
    print(f"{r['item']:<12} {r['weight']:>12.2f} {r['calories']:>18.2f}")

print("-" * 50)

# Total
print(f"{'TOTAL':<12} {'':>12} {total_calories:>18.2f} kcal\n")