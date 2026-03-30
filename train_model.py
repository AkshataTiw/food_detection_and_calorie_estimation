import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("features.csv")
os.makedirs("models", exist_ok=True)

foods = sorted(df["food"].unique())

for food in foods:
    print(f"\n===== Training model for {food} =====")
    df_food = df[df["food"] == food].copy()

    if len(df_food) < 3:
        print("⚠ Not enough data, skipping")
        continue

    X = df_food.drop(columns=["image_name", "weight", "food"])
    y = df_food["weight"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Train MAE:", round(mean_absolute_error(y_train, y_train_pred), 2), "grams")
    print("Test  MAE:", round(mean_absolute_error(y_test, y_test_pred), 2), "grams")
    print("Train R2 :", round(r2_score(y_train, y_train_pred), 4))
    print("Test  R2 :", round(r2_score(y_test, y_test_pred), 4))

    joblib.dump(model, f"models/model_{food}.pkl")
    joblib.dump(X.columns.tolist(), f"models/columns_{food}.pkl")

print("\n✅ All class-wise models trained")