import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

# =========================
# LOAD MODEL
# =========================
model = YOLO("best_new.pt")

# =========================
# PATHS
# =========================
folder = "images"
labels_path = "labels.csv"

# =========================
# READ LABELS
# =========================
df = pd.read_csv(labels_path)
df.columns = df.columns.str.strip().str.lower()

# =========================
# BUILD EXACT FILE MAP
# =========================
filename_to_paths = {}

for root, dirs, files in os.walk(folder):
    for f in files:
        key = f.lower().strip()
        full_path = os.path.join(root, f)

        if key not in filename_to_paths:
            filename_to_paths[key] = []
        filename_to_paths[key].append(full_path)

data = []

for _, row in df.iterrows():
    img_name = str(row["image_name"]).strip()
    weight = float(row["weight_grams"])

    matches = filename_to_paths.get(img_name.lower(), [])

    if len(matches) == 0:
        print(f"❌ File not found: {img_name}")
        continue

    if len(matches) > 1:
        print(f"⚠ Duplicate filename found, skipping for safety: {img_name}")
        for p in matches:
            print("   ", p)
        continue

    path = matches[0]
    food_class = os.path.basename(os.path.dirname(path)).lower().strip()

    results = model(path, conf=0.2)

    for r in results:
        if r.masks is None:
            print(f"❌ No mask detected: {img_name}")
            continue

        masks = r.masks.data.cpu().numpy()

        # keep only single-object images for clean regression
        if len(masks) != 1:
            print(f"⚠ Skipping {img_name} because masks found = {len(masks)}")
            continue

        for m in masks:
            mask = (m > 0.5).astype(np.uint8)

            y_idx, x_idx = np.where(mask)
            if len(x_idx) == 0 or len(y_idx) == 0:
                print(f"⚠ Empty mask: {img_name}")
                continue

            # =========================
            # BASIC FEATURES
            # =========================
            mask_area = int(np.sum(mask))
            img_area = int(mask.shape[0] * mask.shape[1])
            normalized_area = mask_area / img_area if img_area != 0 else 0

            x_min, x_max = x_idx.min(), x_idx.max()
            y_min, y_max = y_idx.min(), y_idx.max()

            width = int(x_max - x_min + 1)
            height = int(y_max - y_min + 1)

            bbox_area = width * height
            aspect_ratio = width / height if height != 0 else 0
            extent = mask_area / bbox_area if bbox_area != 0 else 0

            # =========================
            # SHAPE FEATURES
            # =========================
            labeled = label(mask)
            regions = regionprops(labeled)

            if len(regions) > 0:
                region = max(regions, key=lambda r: r.area)
                perimeter = region.perimeter
                convex_area = region.area_convex
                solidity = mask_area / convex_area if convex_area != 0 else 0
                eccentricity = region.eccentricity
            else:
                perimeter = 0
                solidity = 0
                eccentricity = 0

            # =========================
            # EXTRA FEATURES
            # =========================
            volume_approx = mask_area * (width + height) / 2
            shape_factor = (width * height) / mask_area if mask_area != 0 else 0

            data.append({
                "image_name": img_name,
                "food": food_class,
                "width": width,
                "height": height,
                "bbox_area": bbox_area,
                "mask_area": mask_area,
                "normalized_area": normalized_area,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "perimeter": perimeter,
                "solidity": solidity,
                "eccentricity": eccentricity,
                "volume_approx": volume_approx,
                "shape_factor": shape_factor,
                "weight": weight
            })

# =========================
# SAVE FEATURES
# =========================
df_out = pd.DataFrame(data)
df_out.to_csv("features.csv", index=False)

print("\n✅ features.csv saved")
print("Total rows:", len(df_out))
print("\nFood counts:")
print(df_out["food"].value_counts())