import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("images", exist_ok=True)
df = pd.read_csv("data/listings_clean.csv")
sns.set_theme(style="whitegrid")

# 1. How prices are distributed
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50)
plt.title("Distribution of nightly prices (Berlin Airbnb)")
plt.xlabel("Price per night")
plt.ylabel("Number of listings")
plt.tight_layout()
plt.savefig("images/01_price_distribution.png", dpi=150)
plt.close()

# 2. Price by room type
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="room_type", y="price")
plt.title("Price by room type")
plt.xlabel("Room type")
plt.ylabel("Price per night")
plt.tight_layout()
plt.savefig("images/02_price_by_room_type.png", dpi=150)
plt.close()

# 3. Median price by district
district = (
    df.groupby("neighbourhood_group_cleansed")["price"]
    .median()
    .sort_values()
)
plt.figure(figsize=(8, 6))
district.plot(kind="barh")
plt.title("Median nightly price by district")
plt.xlabel("Median price per night")
plt.ylabel("District")
plt.tight_layout()
plt.savefig("images/03_median_price_by_district.png", dpi=150)
plt.close()

# 4. Price by number of guests
subset = df[df["accommodates"] <= 8]
plt.figure(figsize=(8, 5))
sns.boxplot(data=subset, x="accommodates", y="price")
plt.title("Price by number of guests accommodated")
plt.xlabel("Guests accommodated")
plt.ylabel("Price per night")
plt.tight_layout()
plt.savefig("images/04_price_by_guests.png", dpi=150)
plt.close()

# 5. Correlation heatmap
numeric = df[
    ["price", "accommodates", "bedrooms", "beds", "bathrooms",
     "minimum_nights", "number_of_reviews", "review_scores_rating",
     "availability_365", "host_is_superhost"]
]
corr = numeric.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between numeric features")
plt.tight_layout()
plt.savefig("images/05_correlation_heatmap.png", dpi=150)
plt.close()

print("Median price by room type:")
print(df.groupby("room_type")["price"].median())
print()
print("Median price by district (highest first):")
print(district.sort_values(ascending=False))
print()
print("Correlation with price:")
print(corr["price"].sort_values(ascending=False))
print()
print("Charts saved in the images folder")