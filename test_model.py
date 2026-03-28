import os
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops

# =========================
# LOAD DETECTION MODEL
# =========================
model_det = YOLO("best.pt")

# =========================
# LOAD NUTRITION FILE
# =========================
nutrition_df = pd.read_csv("nutrition.csv")
nutrition_df["food"] = nutrition_df["food"].str.lower().str.strip()
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

# =========================
# LOAD COUNT CONFIG
# =========================
count_df = pd.read_csv("count_based_config.csv")
count_df["food"] = count_df["food"].str.lower().str.strip()
count_dict = dict(zip(count_df["food"], count_df["weight_per_item"]))

# =========================
# TEST IMAGE PATH
# =========================
img_path = "test_images/eggs.jpg"

# =========================
# RUN YOLO
# =========================
results = model_det(img_path, conf=0.2)

total_calories = 0

for r in results:
    if r.masks is None:
        print("❌ No detection")
        continue

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    # ✅ Track processed classes (avoid duplicates)
    processed_classes = set()

    for i, m in enumerate(masks):

        class_id = int(classes[i])
        class_name = model_det.names[class_id].lower().strip()

        # 🚫 Skip if already processed
        if class_name in processed_classes:
            continue

        processed_classes.add(class_name)

        print(f"\nDetected: {class_name}")

        # =========================
        # ✅ COUNT-BASED LOGIC
        # =========================
        if class_name in count_dict:
            count = np.sum(classes == class_id)
            weight_per_item = count_dict[class_name]

            pred_weight = count * weight_per_item

            print("Count detected:", count)
            print("Predicted Weight (count-based):", round(pred_weight, 2), "grams")

        else:
            # =========================
            # ✅ REGRESSION LOGIC (use first mask only)
            # =========================
            mask = (m > 0.5).astype(np.uint8)

            y_idx, x_idx = np.where(mask)
            if len(x_idx) == 0 or len(y_idx) == 0:
                print("⚠ Empty mask, skipping")
                continue

            model_path = f"models/model_{class_name}.pkl"
            cols_path = f"models/columns_{class_name}.pkl"

            if not os.path.exists(model_path) or not os.path.exists(cols_path):
                print(f"⚠ No regression model found for {class_name}")
                continue

            model_reg = joblib.load(model_path)
            model_columns = joblib.load(cols_path)

            # =========================
            # FEATURE EXTRACTION
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
                "eccentricity": eccentricity,
                "volume_approx": volume_approx,
                "shape_factor": shape_factor,
            }

            features_df = pd.DataFrame([features])

            for col in model_columns:
                if col not in features_df.columns:
                    features_df[col] = 0

            features_df = features_df[model_columns]

            pred_weight = model_reg.predict(features_df)[0]
            pred_weight = max(pred_weight, 0)

            print("Predicted Weight (regression):", round(pred_weight, 2), "grams")

        # =========================
        # CALORIE CALCULATION
        # =========================
        if class_name in calorie_dict:
            kcal = (pred_weight / 100.0) * calorie_dict[class_name]
            total_calories += kcal
            print("Calories:", round(kcal, 2), "kcal")
        else:
            print(f"⚠ {class_name} not found in nutrition.csv")

print("\n🍽️ Total Calories:", round(total_calories, 2), "kcal")