import pandas as pd

df = pd.read_csv("data/products.csv")

matches = df[
    df["SKU"].astype(str).str.strip() == "AA201577"
]

for index, row in matches.iterrows():
    print(
        "Row:", index,
        "| Image:", f"data/images/{index:06d}.jpg",
        "| Name:", row["Name"]
    )