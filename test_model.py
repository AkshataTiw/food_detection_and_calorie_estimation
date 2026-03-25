import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops
import joblib
import os

model_det = YOLO("best.pt")

nutrition_df = pd.read_csv("nutrition.csv")
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

img_path = "test_images/test12.jpeg"

results = model_det(img_path, conf=0.2)

total_calories = 0

for r in results:
    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for i, m in enumerate(masks):
        mask = (m > 0.5).astype(np.uint8)

        class_id = int(classes[i])
        class_name = model_det.names[class_id].lower()

        print("\nDetected:", class_name)

        # 🔥 LOAD CORRECT MODEL
        model_path = f"models/model_{class_name}.pkl"
        col_path = f"models/columns_{class_name}.pkl"

        if not os.path.exists(model_path):
            print("⚠️ No model for this class")
            continue

        model = joblib.load(model_path)
        model_columns = joblib.load(col_path)

        # FEATURES
        mask_area = np.sum(mask)
        img_area = mask.shape[0] * mask.shape[1]
        normalized_area = mask_area / img_area

        y_idx, x_idx = np.where(mask)
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

        # 🔥 NEW FEATURES (MUST MATCH TRAINING)
        volume_approx = mask_area * (width + height) / 2
        shape_factor = (width * height) / mask_area if mask_area != 0 else 0

        features = {
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
            "shape_factor": shape_factor
        }

        features_df = pd.DataFrame([features])

        for col in model_columns:
            if col not in features_df:
                features_df[col] = 0

        features_df = features_df[model_columns]

        pred_weight = model.predict(features_df)[0]

        print("✅ Weight:", round(pred_weight, 2), "grams")

        if class_name in calorie_dict:
            kcal = (pred_weight / 100) * calorie_dict[class_name]
            total_calories += kcal
            print("🔥 Calories:", round(kcal, 2))

print("\n🍽️ Total Calories:", round(total_calories, 2))