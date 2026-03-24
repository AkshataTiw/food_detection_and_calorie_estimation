import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops
import joblib

# load models
model_det = YOLO("best.pt")
model_reg = joblib.load("reg_model.pkl")
model_columns = joblib.load("model_columns.pkl")  # 🔥 load columns

# nutrition data
nutrition_df = pd.read_csv("nutrition.csv")
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

img_path = "test_images/test_50.jpg"

results = model_det(img_path, conf=0.2)

total_calories = 0

for r in results:
    if r.masks is None:
        print("❌ No detection")
        continue

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    for i, m in enumerate(masks):
        mask = (m > 0.5).astype(np.uint8)

        class_id = int(classes[i])
        class_name = model_det.names[class_id].lower()

        print("\nDetected:", class_name)

        # ----- FEATURES -----
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

        # base features
        features = {
            "width": width,
            "height": height,
            "bbox_area": bbox_area,
            "mask_area": mask_area,
            "normalized_area": normalized_area,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "perimeter": perimeter,
            "solidity": solidity
        }

        # 🔥 add class column
        features["food_" + class_name] = 1

        features_df = pd.DataFrame([features])

        # 🔥 match training columns
        for col in model_columns:
            if col not in features_df:
                features_df[col] = 0

        features_df = features_df[model_columns]

        # predict
        pred_weight = model_reg.predict(features_df)[0]
        print("✅ Weight:", round(pred_weight, 2), "grams")

        # calories
        if class_name in calorie_dict:
            kcal_per_100g = calorie_dict[class_name]
            calories = (pred_weight / 100) * kcal_per_100g
            total_calories += calories

            print("🔥 Calories:", round(calories, 2))
        else:
            print("⚠️ Not in nutrition.csv")

print("\n🍽️ Total Calories:", round(total_calories, 2), "kcal")