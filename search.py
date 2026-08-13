import os

import torch
from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel

from qdrant_client import QdrantClient


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "patrickjohncyh/fashion-clip"
COLLECTION_NAME = "tailortalk_products"

QUERY_IMAGE = "data/images/000691.jpg"

TOP_K = 5


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# --------------------------------------------------
# QDRANT
# --------------------------------------------------

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected!")


# --------------------------------------------------
# LOAD FASHIONCLIP
# --------------------------------------------------

print("\nLoading FashionCLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("FashionCLIP loaded!")


# --------------------------------------------------
# LOAD QUERY IMAGE
# --------------------------------------------------

print(f"\nQuery image: {QUERY_IMAGE}")

image = Image.open(
    QUERY_IMAGE
).convert("RGB")


# --------------------------------------------------
# CREATE EMBEDDING
# --------------------------------------------------

inputs = processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():

    outputs = model.get_image_features(
        **inputs
    )

if hasattr(outputs, "pooler_output"):
    embedding = outputs.pooler_output
else:
    embedding = outputs


# Normalize
embedding = embedding / embedding.norm(
    p=2,
    dim=-1,
    keepdim=True
)

query_vector = embedding[0].cpu().tolist()


print("\nEmbedding created!")
print("Vector dimensions:", len(query_vector))


# --------------------------------------------------
# SEARCH QDRANT
# --------------------------------------------------

print("\nSearching Qdrant...")

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=TOP_K,
    with_payload=True
).points


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

for rank, result in enumerate(results, start=1):

    payload = result.payload

    print(f"\n#{rank}")
    print("-" * 40)

    print("Score:", round(result.score, 4))
    print("Name:", payload.get("name"))
    print("SKU:", payload.get("sku"))
    print("Stock:", payload.get("stock"))
    print("Retail Price:", payload.get("retail_price"))
    print("Discounted Price:", payload.get("discounted_price"))
    print("Image URL:", payload.get("image_url"))
    print("Website:", payload.get("website_link"))