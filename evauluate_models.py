import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("features.csv")

foods = sorted(df["food"].unique())

results = []

for food in foods:

    df_food = df[df["food"] == food]

    if len(df_food) < 10:
        continue

    X = df_food.drop(columns=["food", "weight", "log_weight"])
    y = df_food["log_weight"]

    # SAME SPLIT (important)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load models
    xgb = joblib.load(f"models/xgb_{food}.pkl")
    rf = joblib.load(f"models/rf_{food}.pkl")

    # Predictions
    pred_xgb = np.exp(xgb.predict(X_test)) - 1
    pred_rf = np.exp(rf.predict(X_test)) - 1

    pred = 0.5 * pred_xgb + 0.5 * pred_rf
    true = np.exp(y_test) - 1

    # Metrics
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((true - pred) / (true + 1e-6))) * 100

    results.append({
        "food": food,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE (%)": mape
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_metrics.csv", index=False)

print(results_df)
print("\n✅ Evaluation completed and saved to evaluation_metrics.csv")