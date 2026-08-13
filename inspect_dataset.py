import pandas as pd

CSV_PATH = "data/products.csv"

df = pd.read_csv(CSV_PATH)

print("Total rows:", len(df))
print("Unique SKUs:", df["SKU"].nunique())

duplicates = df[df["SKU"].duplicated(keep=False)].sort_values("SKU")

print("\nDuplicate SKU rows:")
print(
    duplicates[
        ["SKU", "Name", "image_url"]
    ].to_string(index=False)
)

print("\nNumber of duplicated SKU rows:", len(duplicates))