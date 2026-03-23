import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# load data
df = pd.read_csv("features.csv")

X = df[[
    "width",
    "height",
    "bbox_area",
    "mask_area",
    "normalized_area",
    "aspect_ratio",
    "extent",
    "perimeter",
    "solidity"
]]

y = df["weight"]

# ✅ train-test split (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)
print("MAE (Mean Absolute Error):", mae, "grams")

# save model
joblib.dump(model, "reg_model.pkl")

print("✅ Model saved as reg_model.pkl")