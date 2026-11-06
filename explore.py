import pandas as pd

df = pd.read_csv("data/listings.csv.gz")

print("Rows and columns:", df.shape)
print()
print("Column names:")
print(df.columns.tolist())
print()
print(df.head(3).T)
