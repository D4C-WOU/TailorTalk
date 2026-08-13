import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "tailortalk_products"

info = client.get_collection(COLLECTION_NAME)

print("Collection:", COLLECTION_NAME)
print("Points:", info.points_count)

# Retrieve one actual point
points = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[1],
    with_vectors=True,
    with_payload=True
)

point = points[0]

print("\nFirst point:")
print("ID:", point.id)

print("\nVector length:", len(point.vector))

print("\nFirst 10 vector values:")
print(point.vector[:10])

print("\nPayload:")
print(point.payload)