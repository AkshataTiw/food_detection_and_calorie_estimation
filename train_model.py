import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("features.csv")
os.makedirs("models", exist_ok=True)

foods = sorted(df["food"].unique())

for food in foods:
    print(f"\n===== Training model for {food} =====")

    df_food = df[df["food"] == food]

    if len(df_food) < 10:
        print("⚠ Not enough data")
        continue

    # 🔥 UPDATED FEATURES
    X = df_food[[
        "area",
        "aspect_ratio",
        "extent",
        "solidity",
        "eccentricity",
        "equiv_diameter",
        "volume_proxy",
        "thickness_proxy",
        "roundness",
        "area_x_solidity",
        "area_x_thickness"
    ]]

    # 🔥 LOG TARGET
    y = df_food["log_weight"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 🔥 convert back
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log) - 1
    y_true = np.exp(y_test) - 1

    print("MAE:", round(mean_absolute_error(y_true, y_pred), 2))
    print("R2 :", round(r2_score(y_true, y_pred), 4))

    joblib.dump(model, f"models/model_{food}.pkl")
    joblib.dump(X.columns.tolist(), f"models/columns_{food}.pkl")

print("\n✅ Training complete (FIXED)")