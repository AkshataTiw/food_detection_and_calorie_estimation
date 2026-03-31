import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

df = pd.read_csv("features.csv")
os.makedirs("models", exist_ok=True)

foods = sorted(df["food"].unique())
calibration = []

for food in foods:

    df_food = df[df["food"] == food]

    if len(df_food) < 10:
        continue

    X = df_food.drop(columns=["food", "weight", "log_weight"])
    y = df_food["log_weight"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ XGBoost
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )

    # ✅ Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    joblib.dump(xgb, f"models/xgb_{food}.pkl")
    joblib.dump(rf, f"models/rf_{food}.pkl")
    joblib.dump(X.columns.tolist(), f"models/cols_{food}.pkl")

    # ✅ ENSEMBLE PRED (ONLY TEST SET)
    pred_xgb = np.exp(xgb.predict(X_test)) - 1
    pred_rf = np.exp(rf.predict(X_test)) - 1

    pred = 0.5 * pred_xgb + 0.5 * pred_rf
    true = np.exp(y_test) - 1

    # ✅ CALIBRATION (RIDGE)
    reg = Ridge().fit(pred.reshape(-1,1), true)

    calibration.append({
        "food": food,
        "a": reg.coef_[0],
        "b": reg.intercept_
    })

pd.DataFrame(calibration).to_csv("calibration.csv", index=False)

print("✅ FINAL training done")