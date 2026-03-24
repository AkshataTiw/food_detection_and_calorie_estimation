import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# load data
df = pd.read_csv("features.csv")

# 🔥 convert class to one-hot
df = pd.get_dummies(df, columns=["food"])

# features & target
X = df.drop(columns=["image_name", "weight"])
y = df["weight"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred), "grams")

# save model
joblib.dump(model, "reg_model.pkl")

# 🔥 save column names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Model + columns saved")