import os
import re


image_folder = r"C:\Users\USER\Desktop\food_detection_cal_estimation\images"   # <-- update this

# check if folder exists
if not os.path.exists(image_folder):
    print("❌ Folder path is wrong!")
    exit()

print("✅ Files found in folder:\n")

for file in os.listdir(image_folder):
    print(file)

print("\n--- Processing files ---\n")

for file in os.listdir(image_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, file)

        # extract weight using regex
        match = re.search(r'(\d+)\s*g', file.lower())
        
        if match:
            weight = int(match.group(1))
            print(f"✅ {file} → Weight: {weight}g")
        else:
            print(f"⚠️ Could not extract weight: {file}")