import os

import torch
from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel

from qdrant_client import QdrantClient

from langchain_core.tools import tool


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "patrickjohncyh/fashion-clip"
COLLECTION_NAME = "tailortalk_products"

# Retrieve more candidates first.
# We will rerank them and return only the best 5.
CANDIDATE_K = 20
FINAL_K = 5


# --------------------------------------------------
# MULTI-VIEW WEIGHTS
# --------------------------------------------------

# Full image gets the highest importance because it
# captures the overall saree design.
FULL_WEIGHT = 0.60

# Upper section helps capture blouse/shoulder,
# neckline and upper-body design.
UPPER_WEIGHT = 0.20

# Lower section helps capture border, pallu,
# print and lower-body details.
LOWER_WEIGHT = 0.20


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not QDRANT_URL:
    raise ValueError("QDRANT_URL not found in .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY not found in .env")


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

def create_image_embedding(image: Image.Image):
    """
    Create a normalized FashionCLIP embedding
    from a PIL image.
    """

    image = image.convert("RGB")

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

    # Normalize embedding
    embedding = embedding / embedding.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    return embedding[0].cpu().tolist()


# --------------------------------------------------
# CREATE MULTIPLE IMAGE VIEWS
# --------------------------------------------------

def create_image_views(image_path: str):
    """
    Create multiple visual views of the query image.

    Views:
        1. Full image
        2. Upper section
        3. Lower section

    This helps FashionCLIP pay attention to fine-grained
    saree details such as border, pallu, print and colour.
    """

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    # ----------------------------------------------
    # FULL IMAGE
    # ----------------------------------------------

    full_image = image

    # ----------------------------------------------
    # UPPER SECTION
    # ----------------------------------------------

    upper_image = image.crop(
        (
            0,
            0,
            width,
            int(height * 0.60)
        )
    )

    # ----------------------------------------------
    # LOWER SECTION
    # ----------------------------------------------

    lower_image = image.crop(
        (
            0,
            int(height * 0.40),
            width,
            height
        )
    )

    return {
        "full": full_image,
        "upper": upper_image,
        "lower": lower_image
    }


# --------------------------------------------------
# SEARCH ONE VIEW
# --------------------------------------------------

def search_view(image: Image.Image):
    """
    Search Qdrant using one image view.
    """

    vector = create_image_embedding(image)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=CANDIDATE_K,
        with_payload=True
    ).points

    return results


# --------------------------------------------------
# PRODUCT KEY
# --------------------------------------------------

def get_product_key(result):
    """
    Create a stable key for combining the same product
    returned by multiple visual views.
    """

    payload = result.payload or {}

    # SKU is preferred when available.
    sku = payload.get("sku")

    if sku:
        return str(sku)

    # Fallback to Qdrant point ID.
    return str(result.id)


# --------------------------------------------------
# LANGCHAIN TOOL
# --------------------------------------------------

@tool
def search_similar_products(image_path: str) -> list:
    """
    Find visually similar sarees from the TailorTalk
    product catalogue.

    The search uses FashionCLIP and Qdrant with
    multi-view retrieval. The full image, upper section
    and lower section are searched separately and their
    results are combined to improve fine-grained visual
    matching.

    Args:
        image_path:
            Local path to the user's query image.

    Returns:
        The top 5 visually similar products with
        similarity scores, product information,
        stock and website links.
    """

    # --------------------------------------------------
    # CREATE IMAGE VIEWS
    # --------------------------------------------------

    views = create_image_views(image_path)

    # --------------------------------------------------
    # SEARCH EACH VIEW
    # --------------------------------------------------

    print("Running multi-view visual search...")

    view_results = {}

    for view_name, image in views.items():

        print(
            f"Searching {view_name} view..."
        )

        view_results[view_name] = search_view(
            image
        )

    # --------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------

    candidates = {}

    for view_name, results in view_results.items():

        for result in results:

            key = get_product_key(result)

            if key not in candidates:

                candidates[key] = {
                    "result": result,
                    "scores": {}
                }

            candidates[key]["scores"][
                view_name
            ] = float(result.score)

    # --------------------------------------------------
    # MULTI-VIEW RERANKING
    # --------------------------------------------------

    ranked_candidates = []

    for key, candidate in candidates.items():

        scores = candidate["scores"]

        weighted_score = 0.0
        total_weight = 0.0

        # Full image
        if "full" in scores:

            weighted_score += (
                scores["full"] * FULL_WEIGHT
            )

            total_weight += FULL_WEIGHT

        # Upper section
        if "upper" in scores:

            weighted_score += (
                scores["upper"] * UPPER_WEIGHT
            )

            total_weight += UPPER_WEIGHT

        # Lower section
        if "lower" in scores:

            weighted_score += (
                scores["lower"] * LOWER_WEIGHT
            )

            total_weight += LOWER_WEIGHT

        # Normalize if a product was not found
        # in all three searches.
        if total_weight > 0:

            final_score = (
                weighted_score / total_weight
            )

        else:

            final_score = 0.0

        ranked_candidates.append(
            {
                "result": candidate["result"],
                "score": final_score
            }
        )

    # --------------------------------------------------
    # SORT BY FINAL SCORE
    # --------------------------------------------------

    ranked_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------
    # BUILD FINAL TOP 5
    # --------------------------------------------------

    products = []

    for candidate in ranked_candidates[:FINAL_K]:

        result = candidate["result"]

        payload = result.payload or {}

        stock = max(
            0,
            int(payload.get("stock") or 0)
        )

        products.append(
            {
            "score": round(
                candidate["score"],
                4
            ),

            "name": payload.get(
                "name"
            ),

            "sku": payload.get(
                "sku"
            ),

            "stock": stock,

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
        }
    )

        
    # --------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------

    print(
        f"Retrieved {len(candidates)} unique "
        f"candidates across all views."
    )

    print(
        f"Returning final top {len(products)} products."
    )

    return products