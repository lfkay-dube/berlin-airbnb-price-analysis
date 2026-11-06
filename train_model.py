import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("images", exist_ok=True)
df = pd.read_csv("data/listings_clean.csv")

# Group rare property types into "Other"
top_types = df["property_type"].value_counts().head(10).index
df["property_type"] = df["property_type"].where(
    df["property_type"].isin(top_types), "Other"
)

target = "price"
categorical = ["room_type", "property_type", "neighbourhood_group_cleansed"]
numeric = [c for c in df.columns if c not in categorical + [target]]

X = df[categorical + numeric]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough",
)

models = {
    "Baseline (median)": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

results = []
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        "R2": round(r2_score(y_test, pred), 3),
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results)
print("Model comparison (test set):")
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)

# Which features matter most in the Random Forest?
rf = fitted["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
importance = pd.Series(
    rf.named_steps["model"].feature_importances_, index=names
).sort_values(ascending=False)
importance.index = [n.split("__", 1)[1] for n in importance.index]

print()
print("Top 10 most important features:")
print(importance.head(10).round(3))

plt.figure(figsize=(8, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 features driving price (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/06_feature_importance.png", dpi=150)
plt.close()