import numpy as np
import pandas as pd
from ultralytics import YOLO
from skimage.measure import label, regionprops
import joblib

# load models
model_det = YOLO("best.pt")
model_reg = joblib.load("reg_model.pkl")

# -------- LOAD NUTRITION CSV --------
nutrition_df = pd.read_csv("nutrition.csv")

# create dictionary: food -> kcal_per_100g
calorie_dict = dict(zip(nutrition_df["food"], nutrition_df["kcal_per_100g"]))

img_path = "test_images/test3.jpg"

results = model_det(img_path, conf=0.2)

for r in results:
    if r.masks is None:
        print("❌ No detection")
        exit()

    # -------- CLASS NAME --------
    class_id = int(r.boxes.cls[0])
    class_name = model_det.names[class_id].lower()   # IMPORTANT: lowercase

    print("Detected:", class_name)

    # -------- MASK --------
    mask = r.masks.data.cpu().numpy()[0]
    mask = (mask > 0.5).astype(np.uint8)

    # -------- FEATURES --------
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
    region = regionprops(labeled)[0]

    perimeter = region.perimeter
    solidity = mask_area / region.area_convex if region.area_convex != 0 else 0

    # -------- FEATURE DF --------
    features = pd.DataFrame([{
        "width": width,
        "height": height,
        "bbox_area": bbox_area,
        "mask_area": mask_area,
        "normalized_area": normalized_area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "perimeter": perimeter,
        "solidity": solidity
    }])

    # -------- PREDICT WEIGHT --------
    pred_weight = model_reg.predict(features)[0]

    print("✅ Predicted weight:", round(pred_weight, 2), "grams")

    # -------- CALORIES FROM CSV --------
    if class_name in calorie_dict:
        kcal_per_100g = calorie_dict[class_name]
        calories = (pred_weight / 100) * kcal_per_100g

        print("🔥 Estimated calories:", round(calories, 2), "kcal")
    else:
        print("⚠️ Food not found in nutrition.csv:", class_name)