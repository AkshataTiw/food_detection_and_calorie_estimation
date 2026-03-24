import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

# load model
model = YOLO("best.pt")

folder = "images"
df = pd.read_csv("labels.csv")

df.columns = df.columns.str.strip().str.lower()

data = []

for _, row in df.iterrows():
    img_name = row["image_name"]
    weight = row["weight_grams"]

    path = os.path.join(folder, img_name)

    if not os.path.exists(path):
        print("Missing:", img_name)
        continue

    results = model(path, conf=0.2)

    for r in results:
        if r.masks is None:
            print("No detection:", img_name)
            continue

        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i, m in enumerate(masks):
            mask = (m > 0.5).astype(np.uint8)

            class_id = int(classes[i])
            class_name = model.names[class_id].lower()

            # ----- FEATURES -----
            mask_area = np.sum(mask)
            img_area = mask.shape[0] * mask.shape[1]
            normalized_area = mask_area / img_area

            y_idx, x_idx = np.where(mask)
            x_min, x_max = x_idx.min(), x_idx.max()
            y_min, y_max = y_idx.min(), y_idx.max()

            width = x_max - x_min
            height = y_max - y_min

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

            data.append({
                "image_name": img_name,
                "food": class_name,   # 🔥 IMPORTANT
                "width": width,
                "height": height,
                "bbox_area": bbox_area,
                "mask_area": mask_area,
                "normalized_area": normalized_area,
                "aspect_ratio": aspect_ratio,
                "extent": extent,
                "perimeter": perimeter,
                "solidity": solidity,
                "weight": weight
            })

# save
df_out = pd.DataFrame(data)
df_out.to_csv("features.csv", index=False)

print("✅ Saved features.csv")