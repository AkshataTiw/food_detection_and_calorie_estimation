import os
import re
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

model = YOLO("best.pt")

folder = "images"

df = pd.read_csv("labels.csv")
df.columns = df.columns.str.strip().str.lower()

data = []

all_files = []
for root, dirs, files in os.walk(folder):
    for f in files:
        all_files.append((f, os.path.join(root, f)))

for _, row in df.iterrows():
    img_name = row["image_name"]
    weight = row["weight_grams"]

    match_id = re.search(r'img(\d+)', img_name.lower())
    if not match_id:
        continue

    img_id = match_id.group(1)

    path = None
    for fname, fpath in all_files:
        if f"img{img_id}" in fname.lower():
            path = fpath
            break

    if path is None:
        continue

    food_class = os.path.basename(os.path.dirname(path))

    results = model(path, conf=0.2)

    for r in results:
        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        for m in masks:
            mask = (m > 0.5).astype(np.uint8)

            mask_area = np.sum(mask)
            img_area = mask.shape[0] * mask.shape[1]
            normalized_area = mask_area / img_area

            y_idx, x_idx = np.where(mask)
            if len(x_idx) == 0:
                continue

            width = x_idx.max() - x_idx.min()
            height = y_idx.max() - y_idx.min()

            bbox_area = width * height
            aspect_ratio = width / height if height != 0 else 0
            extent = mask_area / bbox_area if bbox_area != 0 else 0

            labeled = label(mask)
            regions = regionprops(labeled)

            if len(regions) > 0:
                region = regions[0]
                perimeter = region.perimeter
                solidity = mask_area / region.area_convex if region.area_convex != 0 else 0
            else:
                perimeter = 0
                solidity = 0

            # 🔥 NEW FEATURES
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
                "volume_approx": volume_approx,
                "shape_factor": shape_factor,
                "weight": weight
            })

pd.DataFrame(data).to_csv("features.csv", index=False)
print("✅ features.csv saved")