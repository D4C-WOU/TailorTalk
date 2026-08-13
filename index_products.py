import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "patrickjohncyh/fashion-clip"
COLLECTION_NAME = "tailortalk_products"

CSV_PATH = "data/products.csv"
IMAGE_DIR = "data/images"

BATCH_SIZE = 8


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# --------------------------------------------------
# CONNECT TO QDRANT
# --------------------------------------------------

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected!")


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

print("\nLoading CSV...")

df = pd.read_csv(CSV_PATH)
df.columns = (
    df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

print("Columns:", [repr(c) for c in df.columns])

print(f"Total products in CSV: {len(df)}")
print("Columns:", list(df.columns))

# Handle both possible website column names
if "WebsiteLink" in df.columns:
    website_column = "WebsiteLink"
elif "Website Link" in df.columns:
    website_column = "Website Link"
else:
    website_column = None

print("Website column:", website_column)


# --------------------------------------------------
# LOAD FASHIONCLIP
# --------------------------------------------------

print("\nLoading FashionCLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("FashionCLIP loaded!")


# --------------------------------------------------
# PREPARE PRODUCTS
# --------------------------------------------------

products = []
missing_images = []

for index, row in df.iterrows():

    # Image filename is based on CSV row index
    image_filename = f"{index:06d}.jpg"

    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )

    if not os.path.exists(image_path):
        missing_images.append({
            "index": index,
            "sku": str(row["SKU"]).strip()
        })
        continue

    products.append({
        "id": index + 1,
        "image_path": image_path,
        "payload": {
            "name": None if pd.isna(row["Name"])
                else str(row["Name"]),

            "sku": str(row["SKU"]).strip(),

            "stock": None if pd.isna(row["Stock"])
                else row["Stock"],

            "retail_price": None
                if pd.isna(row["Retail Price"])
                else row["Retail Price"],

            "discounted_price": None
                if pd.isna(row["Discounted Price"])
                else row["Discounted Price"],

            "image_url": None
                if pd.isna(row["image_url"])
                else str(row["image_url"]),

            "website_link": (
                None
                if website_column is None or pd.isna(row[website_column])
                else str(row[website_column])),
        }
    })


print(f"\nProducts with images: {len(products)}")
print(f"Missing images: {len(missing_images)}")


# --------------------------------------------------
# GENERATE EMBEDDINGS + UPLOAD
# --------------------------------------------------

print("\nGenerating embeddings and uploading to Qdrant...")

for start in tqdm(
    range(0, len(products), BATCH_SIZE),
    desc="Indexing"
):

    batch = products[
        start:start + BATCH_SIZE
    ]

    images = []

    for product in batch:

        image = Image.open(
            product["image_path"]
        ).convert("RGB")

        images.append(image)

    # Process batch
    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    # Generate embeddings
    with torch.no_grad():

        outputs = model.get_image_features(
            **inputs
        )

    # Handle HuggingFace output
    if hasattr(outputs, "pooler_output"):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs

    # Normalize
    embeddings = embeddings / embeddings.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    embeddings = embeddings.cpu().tolist()

    # Create Qdrant points
    points = []

    for product, vector in zip(
        batch,
        embeddings
    ):

        points.append(
            PointStruct(
                id=product["id"],
                vector=vector,
                payload=product["payload"]
            )
        )

    # Upload batch
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )


# --------------------------------------------------
# VERIFY
# --------------------------------------------------

print("\nIndexing complete!")

collection_info = client.get_collection(
    COLLECTION_NAME
)

print(
    f"Vectors stored in Qdrant: "
    f"{collection_info.points_count}"
)

print(
    f"Missing images: "
    f"{len(missing_images)}"
)

if missing_images:

    print("\nMissing rows:")

    for item in missing_images:

        print(
            f"- Row {item['index']} "
            f"| SKU {item['sku']}"
        )