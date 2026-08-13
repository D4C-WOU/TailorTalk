from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

MODEL_NAME = "patrickjohncyh/fashion-clip"

print("Loading FashionCLIP...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

image_path = "data/images/" + os.listdir("data/images")[0]

print("Testing image:", image_path)

image = Image.open(image_path).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.get_image_features(**inputs)

# Handle the output returned by the installed Transformers version
if hasattr(outputs, "pooler_output"):
    image_features = outputs.pooler_output
else:
    image_features = outputs

# Normalize the embedding
image_features = image_features / image_features.norm(
    p=2,
    dim=-1,
    keepdim=True
)

print("Embedding shape:", image_features.shape)
print("First 10 values:")
print(image_features[0][:10])