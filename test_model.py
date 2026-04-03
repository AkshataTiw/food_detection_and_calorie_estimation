import os
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

import warnings
warnings.filterwarnings("ignore")

# =========================
# LOAD MODELS & FILES
# =========================
model_det = YOLO("best_new.pt")

calib_df = pd.read_csv("calibration.csv")

nutrition_df = pd.read_csv("nutrition.csv")
nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

# ✅ COUNT CONFIG
count_df = pd.read_csv("count_based_config.csv")
count_df["food"] = count_df["food"].str.lower().str.strip()
count_weight_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

# =========================
# INPUT IMAGE
# =========================
img_path = "test_images2/test43_cuc115_capsi_160_t95.jpeg"

results = model_det(img_path, conf=0.25)

total_calories = 0

print("\n🍽️ Food Prediction Summary")
print("=" * 65)

print(f"{'S.No':<6}{'Item':<20}{'Weight (g)':>15}{'Calories (kcal)':>20}")
print("-" * 65)

rows = []
count = 1

# 🔥 STORE COUNTS
count_items = {}

# =========================
# MAIN LOOP
# =========================
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

        # =========================
        # ✅ COUNT-BASED (STORE ONLY)
        # =========================
        if food in count_weight_dict:
            count_items[food] = count_items.get(food, 0) + 1
            continue

        # =========================
        # 🔥 REGRESSION (UNCHANGED)
        # =========================
        xgb = joblib.load(f"models/xgb_{food}.pkl")
        rf = joblib.load(f"models/rf_{food}.pkl")
        cols = joblib.load(f"models/cols_{food}.pkl")

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

        # CALIBRATION
        row = calib_df[calib_df["food"] == food]
        if len(row) > 0:
            pred = row["a"].values[0] * pred + row["b"].values[0]

        kcal = (pred / 100) * calorie_dict.get(food, 0)
        total_calories += kcal

        rows.append((count, food, pred, kcal))
        count += 1


# =========================
# 🔥 PROCESS COUNT ITEMS
# =========================
for food, cnt in count_items.items():
    weight_per_item = count_weight_dict[food]

    total_weight = cnt * weight_per_item
    kcal = (total_weight / 100) * calorie_dict.get(food, 0)

    total_calories += kcal

    rows.append((count, f"{food} x {cnt}", total_weight, kcal))
    count += 1


# =========================
# PRINT RESULTS
# =========================
for row in rows:
    print(f"{row[0]:<6}{row[1]:<20}{row[2]:>15.2f}{row[3]:>20.2f}")

print("-" * 65)

print(f"{'TOTAL':<21}{'':>15}{total_calories:>20.2f}")

print("=" * 65)