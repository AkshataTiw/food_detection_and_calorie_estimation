import os
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

import warnings
warnings.filterwarnings("ignore")

model_det = YOLO("best_new.pt")

calib_df = pd.read_csv("calibration.csv")

nutrition_df = pd.read_csv("nutrition.csv")
nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

img_path = "test_images2/test11_a100_po145.jpeg"

results = model_det(img_path, conf=0.25)

total_calories = 0

print("\n🍽️ Food Prediction Summary")
print("=" * 60)

# Table Header
print(f"{'S.No':<6}{'Item':<15}{'Weight (g)':>15}{'Calories (kcal)':>20}")
print("-" * 60)

rows = []
count = 1

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

        xgb = joblib.load(f"models/xgb_{food}.pkl")
        rf = joblib.load(f"models/rf_{food}.pkl")
        cols = joblib.load(f"models/cols_{food}.pkl")

        # FEATURES (unchanged)
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


# 🔥 PRINT TABLE ROWS
for row in rows:
    print(f"{row[0]:<6}{row[1]:<15}{row[2]:>15.2f}{row[3]:>20.2f}")

print("-" * 60)

# 🔥 TOTAL ROW
print(f"{'TOTAL':<21}{'':>15}{total_calories:>20.2f}")

print("=" * 60)