import pandas as pd

df = pd.read_csv("data/listings.csv.gz")

columns = [
    "price", "room_type", "property_type", "neighbourhood_group_cleansed",
    "accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights",
    "number_of_reviews", "review_scores_rating", "availability_365",
    "host_is_superhost", "instant_bookable", "latitude", "longitude",
]
df = df[columns].copy()

print("Raw price examples:")
print(df["price"].head(10).tolist())

df["price"] = pd.to_numeric(
    df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce",
)

print()
print("Missing values per column:")
print(df.isna().sum())

print()
print("Price summary:")
print(df["price"].describe())

print()
print("Listings per room type:")
print(df["room_type"].value_counts())