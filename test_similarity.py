from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

MODEL_NAME = "patrickjohncyh/fashion-clip"

print("Loading FashionCLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

image_dir = "data/images"

# Take first 5 images
image_files = os.listdir(image_dir)[:5]

images = [
    Image.open(os.path.join(image_dir, filename)).convert("RGB")
    for filename in image_files
]

inputs = processor(
    images=images,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.get_image_features(**inputs)

if hasattr(outputs, "pooler_output"):
    embeddings = outputs.pooler_output
else:
    embeddings = outputs

# Normalize embeddings
embeddings = embeddings / embeddings.norm(
    p=2,
    dim=-1,
    keepdim=True
)

# Cosine similarity = dot product because vectors are normalized
similarity_matrix = embeddings @ embeddings.T

print("\nImages:")
for i, filename in enumerate(image_files):
    print(i, filename)

print("\nSimilarity Matrix:")
print(similarity_matrix)