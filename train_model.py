import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv("features.csv")

foods = df["food"].unique()

os.makedirs("models", exist_ok=True)

for food in foods:
    print(f"\n===== Training for {food} =====")

    df_food = df[df["food"] == food]

    if len(df_food) < 10:
        print("⚠️ Not enough data, skipping")
        continue

    X = df_food.drop(columns=["image_name", "weight", "food"])
    y = df_food["weight"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    print("MAE:", round(mae, 2), "grams")

    # save
    joblib.dump(model, f"models/model_{food}.pkl")
    joblib.dump(X.columns.tolist(), f"models/columns_{food}.pkl")

print("\n✅ All models trained")