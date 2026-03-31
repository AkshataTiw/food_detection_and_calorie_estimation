import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

model = YOLO("best_new.pt")

folder = "images"
labels_path = "labels.csv"

df = pd.read_csv(labels_path)
df.columns = df.columns.str.strip().str.lower()

filename_to_paths = {}
for root, _, files in os.walk(folder):
    for f in files:
        filename_to_paths[f.lower()] = os.path.join(root, f)

data = []

for _, row in df.iterrows():
    img_name = str(row["image_name"]).strip()
    weight = float(row["weight_grams"])

    if img_name.lower() not in filename_to_paths:
        continue

    path = filename_to_paths[img_name.lower()]
    food_class = os.path.basename(os.path.dirname(path)).lower().strip()

    results = model(path, conf=0.25)

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        # ✅ TAKE LARGEST OBJECT ONLY
        areas = [np.sum(m > 0.5) for m in masks]
        if len(areas) == 0:
            continue

        mask = (masks[np.argmax(areas)] > 0.5).astype(np.uint8)

        y_idx, x_idx = np.where(mask)
        if len(x_idx) == 0:
            continue

        # ✅ CROP
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
        convex_area = region.convex_area
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length

        # =========================
        # 🔥 FINAL FEATURES
        # =========================

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

        data.append({
            "food": food_class,

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

            "weight": weight
        })

df_out = pd.DataFrame(data)

# ✅ CLEAN OUTLIERS
df_out = df_out[(df_out["weight"] > 5) & (df_out["weight"] < 500)]

# ✅ LOG TARGET
df_out["log_weight"] = np.log(df_out["weight"] + 1)

df_out.to_csv("features.csv", index=False)

print("✅ FINAL features.csv ready")