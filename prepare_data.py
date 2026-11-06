import pandas as pd

df = pd.read_csv("data/listings.csv.gz")

columns = [
    "price", "room_type", "property_type", "neighbourhood_group_cleansed",
    "accommodates", "bedrooms", "beds", "bathrooms_text", "minimum_nights",
    "number_of_reviews", "review_scores_rating", "availability_365",
    "host_is_superhost", "latitude", "longitude",
]
df = df[columns].copy()

# 1. Price: text -> number
df["price"] = pd.to_numeric(
    df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce",
)

# 2. Drop rows without a price
before = len(df)
df = df.dropna(subset=["price"])
print(f"Rows dropped (no price): {before - len(df)}; rows left: {len(df)}")

# 3. Bathrooms: take the number from text such as "1 bath" or "1.5 shared baths"
df["bathrooms"] = pd.to_numeric(
    df["bathrooms_text"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
)
half = df["bathrooms_text"].str.contains("half", case=False, na=False)
df.loc[half & df["bathrooms"].isna(), "bathrooms"] = 0.5
df = df.drop(columns=["bathrooms_text"])

# 4. Superhost: t/f -> 1/0
df["host_is_superhost"] = df["host_is_superhost"].map({"t": 1, "f": 0})

# 5. Rare room types -> "Other"
df["room_type"] = df["room_type"].replace(
    {"Shared room": "Other", "Hotel room": "Other"}
)

# 6. Reviews: add a flag, then fill missing ratings with the median
df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)
df["review_scores_rating"] = df["review_scores_rating"].fillna(
    df["review_scores_rating"].median()
)

# 7. Fill small gaps with the median
for col in ["bedrooms", "beds", "bathrooms", "minimum_nights"]:
    df[col] = df[col].fillna(df[col].median())

# 8. Remove extreme prices (above the 99th percentile)
cap = df["price"].quantile(0.99)
print(f"99th percentile price: {cap:.2f}")
df = df[df["price"] <= cap]
print("Rows after removing extreme prices:", len(df))

print()
print("Superhost values:")
print(df["host_is_superhost"].value_counts(dropna=False))
print()
print("Missing values after cleaning:")
print(df.isna().sum())
print()
print(df["price"].describe())

df.to_csv("data/listings_clean.csv", index=False)
print()
print("Saved data/listings_clean.csv")