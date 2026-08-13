import os

import torch
from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from langchain_core.tools import tool


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "patrickjohncyh/fashion-clip"
COLLECTION_NAME = "tailortalk_products"

TOP_K = 5


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading FashionCLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("FashionCLIP loaded!")


# --------------------------------------------------
# QDRANT
# --------------------------------------------------

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant!")


# --------------------------------------------------
# IMAGE EMBEDDING
# --------------------------------------------------

def create_image_embedding(image_path: str):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.get_image_features(
            **inputs
        )

    # HuggingFace compatibility
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

    return embedding[0].cpu().tolist()


# --------------------------------------------------
# LANGCHAIN TOOL
# --------------------------------------------------

@tool
def search_similar_products(image_path: str) -> list:
    """
    Find visually similar sarees from the TailorTalk product catalogue.

    Use this tool when the user provides an image and wants to find
    visually similar clothing products.

    Args:
        image_path: Local path to the user's query image.

    Returns:
        A list of the closest matching products with similarity scores,
        product information, and website links.
    """

    # Create query embedding
    vector = create_image_embedding(image_path)

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=TOP_K,
        with_payload=True
    ).points

    products = []

    for result in results:

        payload = result.payload or {}

        products.append({
            "score": round(float(result.score), 4),

            "name": payload.get("name"),

            "sku": payload.get("sku"),

            "stock": payload.get("stock"),

            "retail_price": payload.get(
                "retail_price"
            ),

            "discounted_price": payload.get(
                "discounted_price"
            ),

            "image_url": payload.get(
                "image_url"
            ),

            "website_link": payload.get(
                "website_link"
            )
        })

    return products