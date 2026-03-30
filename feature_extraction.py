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

        # 🔥 IMPORTANT: take LARGEST mask only (fix multi-object noise)
        areas = [np.sum(m > 0.5) for m in masks]
        if len(areas) == 0:
            continue

        mask = (masks[np.argmax(areas)] > 0.5).astype(np.uint8)

        y_idx, x_idx = np.where(mask)
        if len(x_idx) == 0:
            continue

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
        convex_area = region.convex_area

        width = mask_crop.shape[1]
        height = mask_crop.shape[0]

        # =========================
        # 🔥 NEW ROBUST FEATURES
        # =========================

        area = mask_area
        aspect_ratio = width / (height + 1e-6)

        extent = area / (width * height + 1e-6)
        solidity = area / (convex_area + 1e-6)
        eccentricity = region.eccentricity

        # 🔥 KEY FIX: Equivalent diameter (stable volume base)
        equiv_diameter = np.sqrt(4 * area / np.pi)

        # 🔥 TRUE VOLUME PROXY (major fix)
        volume_proxy = equiv_diameter ** 3

        # 🔥 THICKNESS (bbox based, NOT perimeter)
        thickness_proxy = area / (width * height + 1e-6)

        # 🔥 ROUNDNESS
        roundness = (4 * np.pi * area) / (perimeter**2 + 1e-6)

        # 🔥 INTERACTIONS (VERY IMPORTANT)
        area_x_solidity = area * solidity
        area_x_thickness = area * thickness_proxy

        data.append({
            "food": food_class,

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
            "area_x_thickness": area_x_thickness,

            "weight": weight
        })

df_out = pd.DataFrame(data)

# 🔥 LOG TARGET (CRITICAL FIX)
df_out["log_weight"] = np.log(df_out["weight"] + 1)

df_out.to_csv("features.csv", index=False)

print("✅ features.csv ready (FIXED)")