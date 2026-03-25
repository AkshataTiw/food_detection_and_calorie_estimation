import pandas as pd

df = pd.read_csv("features.csv")

cucumber_df = df[df["food"] == "cucumber"].copy()

print("Total cucumber samples:", len(cucumber_df))

print("\nUnique cucumber weights:")
print(sorted(cucumber_df["weight"].unique()))

print("\nCounts per cucumber weight:")
print(cucumber_df["weight"].value_counts().sort_index())