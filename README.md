# TailorTalk 👗

AI-powered visual fashion search for finding visually similar sarees
from a fashion catalogue.

## Overview

TailorTalk is an AI agent that accepts a fashion image, converts it into
a visual embedding, searches a vector database for visually similar
catalogue products, and presents the closest matches with similarity
scores and product information.

The project is designed around the core requirement of the TailorTalk
assignment: visually similar results should be close in **colour,
fabric, pattern, border/pallu work, and overall design**, rather than
simply returning generic sarees.

## Architecture

``` text
User Image
    │
    ▼
Streamlit Frontend
    │
    ▼
TailorTalk Agent (LangChain + Groq)
    │
    │ function/tool call
    ▼
search_similar_products(image_path)
    │
    ▼
FashionCLIP
    │
    │ 512-dimensional normalized embedding
    ▼
Qdrant Vector Database
    │
    │ Top-K similarity search
    ▼
Product Metadata
    │
    ▼
Groq LLM
    │
    ▼
TailorTalk Response
    │
    ├── Product name
    ├── Similarity score
    ├── Price
    ├── Stock status
    └── Product link
```

## Technology Stack

  ------------------------------------------------------------------------
  Component               Technology               Purpose
  ----------------------- ------------------------ -----------------------
  Frontend                Streamlit                Web interface and image
                                                   upload

  Agent framework         LangChain                Tool definition and
                                                   function calling

  LLM                     Groq                     Agent reasoning and
                          (`openai/gpt-oss-20b`)   response generation

  Vision model            FashionCLIP              Fashion-specific image
                                                   embeddings

  Vector database         Qdrant                   Fast similarity search
                                                   over embeddings

  Image processing        Pillow                   Image loading and RGB
                                                   conversion

  ML runtime              PyTorch                  FashionCLIP inference

  Environment management  python-dotenv            API key/configuration
                                                   management
  ------------------------------------------------------------------------

## How the Search Works

### 1. Catalogue processing

The provided saree image catalogue is processed and indexed before
running the application.

Each catalogue image is passed through FashionCLIP to create a visual
embedding.

### 2. Embedding generation

FashionCLIP produces a **512-dimensional** image representation.

The embedding is L2-normalized before being stored/searched, allowing
similarity to be compared consistently.

### 3. Vector search

The uploaded query image follows the same embedding pipeline.

The resulting vector is sent to Qdrant, which returns the closest
catalogue vectors using similarity search.

TailorTalk currently retrieves the **top 5** matches.

### 4. Agent tool

The search logic is exposed to LangChain as:

``` python
search_similar_products(image_path: str) -> list
```

The tool returns structured product information including:

-   similarity score
-   product name
-   SKU
-   stock
-   retail price
-   discounted price
-   image URL
-   website URL

The LLM decides when the visual search tool should be called and then
turns the tool output into a concise user-facing recommendation.

## Search Quality

Search quality is the most important part of this project because the
catalogue contains the same broad garment category. Generic visual
similarity is therefore not enough.

The implementation uses **FashionCLIP** rather than a generic image
embedding model because the task is specifically fashion-oriented.

The retrieval pipeline was repeatedly tested with different saree
images. The returned matches were visually inspected for similarity in:

-   colour combinations
-   overall silhouette
-   fabric appearance
-   print/pattern
-   border design
-   pallu design
-   overall styling

The application returns five candidates so users can compare several
close alternatives rather than relying on a single result.

## Agent / Tool Design

The search operation is isolated behind a callable LangChain tool.

This keeps the responsibilities separated:

``` text
LLM
 └── decides whether visual search is required

Tool
 └── performs deterministic image → embedding → Qdrant search

Qdrant
 └── retrieves nearest catalogue products

LLM
 └── explains/recommends the returned products
```

This also prevents the LLM from inventing catalogue results because
product retrieval is performed by the vector-search tool.

## Project Structure

``` text
TailorTalk/
│
├── app.py                  # Streamlit application
├── agent.py                # LangChain/Groq agent and tool orchestration
├── search_tool.py          # FashionCLIP + Qdrant search tool
├── test_tool.py            # Tool-level testing
├── search.py               # Direct vector-search testing
├── data/
│   └── images/             # Local query/test images
├── .env                    # Local secrets (DO NOT COMMIT)
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment Variables

Create a `.env` file:

``` env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or expose API keys publicly.

## Local Setup

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd TailorTalk
```

### 2. Create and activate a virtual environment

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` using the variables shown above.

### 5. Run the Streamlit application

``` bash
python -m streamlit run app.py
```

The application will be available at:

``` text
http://localhost:8501
```

## Testing

### Direct vector search

``` bash
python search.py
```

### LangChain tool test

``` bash
python test_tool.py
```

### Agent test

``` bash
python agent.py
```

## Assumptions

-   The catalogue consists primarily of sarees, so the search task is
    scoped to this fashion category.
-   The input image is expected to contain a reasonably clear view of
    the clothing item.
-   Visual similarity is based on the image representation learned by
    FashionCLIP; it does not guarantee exact product identity.
-   Product stock and pricing are taken from the indexed catalogue
    metadata and may change on the source website.

## Trade-offs

### FashionCLIP

**Advantage:** Fashion-specific embeddings are better suited to clothing
similarity than generic image embeddings.

**Trade-off:** The model adds a relatively heavy local inference
dependency and increases application startup time.

### Qdrant

**Advantage:** Provides a dedicated vector-search layer and scales
better than performing a full in-memory comparison for every query.

**Trade-off:** The application depends on an external vector database
service and therefore requires Qdrant configuration.

### Groq + LangChain

**Advantage:** Keeps the agent/tool-calling layer simple while providing
natural-language responses.

**Trade-off:** An external LLM API is required for the agent response
layer.

### Top-5 retrieval

Returning five candidates gives users alternatives and makes the search
easier to evaluate.

The trade-off is that lower-ranked results may be less visually similar
than the top result.

## Current Features

-   ✅ Fashion image upload
-   ✅ FashionCLIP image embeddings
-   ✅ Qdrant vector search
-   ✅ Top-5 visual similarity retrieval
-   ✅ LangChain callable search tool
-   ✅ Groq function/tool calling
-   ✅ Similarity scores
-   ✅ Price and discount information
-   ✅ Stock availability
-   ✅ Product website links
-   ✅ Product images in the Streamlit results UI
-   ✅ Repeated testing with multiple saree images

## Deployment

The final application is intended to be deployed on a platform such as:

-   Streamlit Community Cloud
-   Hugging Face Spaces
-   Render

Deployment must provide a public application URL and configure the
required environment variables as platform secrets.

## Assignment Alignment

TailorTalk implements the assignment's core pipeline:

``` text
Image
  ↓
FashionCLIP embedding
  ↓
Qdrant vector index
  ↓
LangChain callable tool
  ↓
Groq agent
  ↓
Streamlit interface
  ↓
Top visually similar catalogue products
```

The final submission should include:

1.  A working deployed application URL
2.  The GitHub repository URL
3.  This README with setup instructions, architecture, quality
    improvements, assumptions, and trade-offs
