import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

CSV_PATH = "data/products.csv"
IMAGE_DIR = "data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("Total products:", len(df))
print("Columns:", list(df.columns))

success = 0
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    sku = str(row["SKU"])
    url = row["image_url"]

    if pd.isna(url):
        failed += 1
        continue

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")

        image.save(
            os.path.join(IMAGE_DIR, f"{sku}.jpg"),
            "JPEG"
        )

        success += 1

    except Exception as e:
        failed += 1

print("\nDone!")
print("Downloaded:", success)
print("Failed:", failed)