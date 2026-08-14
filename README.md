# TailorTalk 👗

AI-powered visual fashion search for finding visually similar sarees
from a fashion catalogue.

## Overview

TailorTalk is an AI-powered visual fashion search application that
accepts a fashion image, generates a visual embedding, searches a vector
database for visually similar catalogue products, and presents the
closest matches with similarity scores and product information.

The system is designed around the core requirement of the TailorTalk
assignment: visually similar results should be close in **colour,
fabric, pattern, border/pallu work, and overall design**, rather than
simply returning generic sarees.

## Live Demo

**Deployed Application:**\
https://tailortalk-demo.streamlit.app/

------------------------------------------------------------------------

## Architecture

``` text
User Image
    │
    ▼
Streamlit Frontend
    │
    ▼
TailorTalk Search Pipeline
    │
    ▼
Multi-View Image Processing
    │
    ├── Full Image
    ├── Upper Section
    └── Lower Section
    │
    ▼
FashionCLIP
    │
    │ 512-dimensional normalized embeddings
    ▼
Qdrant Vector Database
    │
    │ Top-20 candidates per view
    ▼
Candidate Combination
    │
    ▼
Weighted Multi-View Reranking
    │
    │ Full: 60%
    │ Upper: 20%
    │ Lower: 20%
    ▼
Final Top-5 Products
    │
    ▼
Groq LLM
    │
    ▼
TailorTalk Response
```

The visual retrieval itself is deterministic: the uploaded image is
processed using FashionCLIP and searched against Qdrant. The retrieved
product data is then passed to the Groq LLM, which generates a concise
natural-language shopping response.

------------------------------------------------------------------------

## Technology Stack

  ------------------------------------------------------------------------
  Component               Technology               Purpose
  ----------------------- ------------------------ -----------------------
  Frontend                Streamlit                Web interface and image
                                                   upload

  Agent / orchestration   LangChain                Tool definition and LLM
                                                   integration

  LLM                     Groq                     Natural-language
                          (`openai/gpt-oss-20b`)   shopping responses

  Vision model            FashionCLIP              Fashion-specific image
                                                   embeddings

  Vector database         Qdrant                   Similarity search over
                                                   product embeddings

  Image processing        Pillow                   Image loading,
                                                   conversion and cropping

  ML runtime              PyTorch                  FashionCLIP inference

  Environment management  python-dotenv            API key and
                                                   configuration
                                                   management
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## How the Search Works

### 1. Catalogue processing

The provided saree image catalogue is processed and indexed before
running the application.

Each catalogue image is passed through FashionCLIP to generate a visual
embedding, which is stored in Qdrant together with the relevant product
metadata.

### 2. Query image processing

When a user uploads an image, TailorTalk creates three visual views:

-   **Full image**
-   **Upper section** --- top 60% of the image
-   **Lower section** --- bottom 60% of the image

The upper and lower regions overlap slightly so that important visual
details around the middle of the garment are not completely excluded.

### 3. Embedding generation

Each image view is passed through FashionCLIP.

The resulting image representation is L2-normalized before being used
for similarity search, allowing the embeddings to be compared
consistently.

### 4. Vector search

Each visual view is independently searched against the Qdrant catalogue.

TailorTalk retrieves up to **20 candidates per view** rather than
immediately returning only the first five results.

### 5. Candidate combination and reranking

Products returned from multiple views are combined using their SKU as
the preferred stable identifier, with the Qdrant point ID used as a
fallback.

A weighted similarity score is then calculated:

``` text
Final Score =
    Full Image Score × 0.60
  + Upper Section Score × 0.20
  + Lower Section Score × 0.20
```

If a product is not returned by one of the views, the available scores
are normalized using the weights that were actually present.

The candidates are then sorted by the final score and the **top 5
products** are returned.

### 6. LLM response generation

The retrieved product information is passed to the Groq LLM.

The LLM generates a concise response using only the retrieved catalogue
data, including information such as:

-   Product name
-   Similarity score
-   Price
-   Stock availability
-   Product link

The LLM is instructed not to invent products, prices, stock information,
scores or URLs.

------------------------------------------------------------------------

## Result Quality Improvements

Search quality is the most important part of this project because the
catalogue contains products from the same broad garment category. A
simple nearest-neighbour search over the complete image may therefore
return products that are generally similar but miss important
fine-grained details.

Several techniques were implemented to improve retrieval quality.

### 1. Fashion-specific embeddings

Instead of using a generic image embedding model, TailorTalk uses
**FashionCLIP**, which is designed for fashion-related visual
representations.

This helps the system capture clothing-specific characteristics such as:

-   Colour
-   Pattern
-   Fabric appearance
-   Garment structure
-   Overall visual style

### 2. Multi-view retrieval

A single embedding of the complete image may not sufficiently emphasize
fine-grained saree details.

To address this, the query image is searched using three different
views:

``` text
                Query Image
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Full       Upper       Lower
          │          │           │
          ▼          ▼           ▼
       FashionCLIP / FashionCLIP / FashionCLIP
          │          │           │
          ▼          ▼           ▼
       Qdrant     Qdrant      Qdrant
```

The different views help the system pay attention to both the overall
design and localized details such as blouse/shoulder design, borders,
pallu and patterns.

### 3. Candidate expansion

Instead of searching for only five products, TailorTalk retrieves up to
**20 candidates for each visual view**.

This creates a larger candidate pool before the final ranking stage.

### 4. Weighted reranking

The candidates from all views are combined and reranked using weighted
similarity scores:

-   **Full image:** 60%
-   **Upper section:** 20%
-   **Lower section:** 20%

The full image receives the highest weight because it captures the
overall appearance of the saree, while the upper and lower sections
provide additional fine-grained visual information.

This approach allows a product that is consistently similar across
multiple views to rank highly rather than relying on a single visual
crop.

### 5. Result inspection and testing

The retrieval pipeline was tested with multiple saree images and the
returned products were visually inspected for similarity in:

-   Colour combinations
-   Overall silhouette
-   Fabric appearance
-   Print and pattern
-   Border design
-   Pallu design
-   Overall styling

The goal was to prioritize meaningful visual similarity rather than
simply returning products belonging to the same broad category.

------------------------------------------------------------------------

## Agent / Tool Design

The search operation is isolated behind a callable LangChain tool:

``` python
search_similar_products(image_path: str) -> list
```

The responsibilities are separated as follows:

``` text
Uploaded Image
      │
      ▼
Application
      │
      ▼
LangChain Search Tool
      │
      ▼
FashionCLIP
      │
      ▼
Qdrant
      │
      ▼
Top-5 Product Results
      │
      ▼
Groq LLM
      │
      ▼
Natural-Language Response
```

The visual search tool performs the actual product retrieval. The LLM is
used after retrieval to explain and recommend the returned products.

This separation prevents the LLM from generating or inventing catalogue
results because product information comes directly from the
vector-search pipeline.

------------------------------------------------------------------------

## Project Structure

``` text
TailorTalk/
│
├── app.py                  # Streamlit application
├── agent.py                # Groq LLM integration and response generation
├── search_tool.py          # FashionCLIP + Qdrant visual search
├── create_collection.py    # Qdrant collection creation
├── index_products.py       # Catalogue embedding and indexing
├── find_query.py            # Query/search testing
├── search.py                # Direct vector-search testing
├── test_tool.py             # LangChain tool testing
├── test_embedding.py        # Embedding testing
├── test_images.py           # Image testing
├── test_qdrant.py           # Qdrant connectivity/testing
├── test_similarity.py       # Similarity testing
├── verify_index.py          # Index verification
├── inspect_dataset.py       # Dataset inspection
├── reset_collection.py      # Qdrant collection reset utility
│
├── data/
│   └── images/              # Local query/test images
│
├── .env                     # Local secrets (DO NOT COMMIT)
├── .gitignore
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Environment Variables

Create a `.env` file locally:

``` env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or expose API keys publicly.

For deployment, configure these values using the deployment platform's
secrets/environment-variable system.

------------------------------------------------------------------------

## Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/D4C-WOU/TailorTalk.git
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

Create a `.env` file using:

``` env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the application

``` bash
python -m streamlit run app.py
```

The application will be available at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

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

Additional testing utilities are included in the repository for checking
embeddings, image processing, Qdrant connectivity, similarity results
and index integrity.

------------------------------------------------------------------------

## Assumptions

-   The catalogue consists primarily of sarees, so the search task is
    scoped to this fashion category.
-   The input image is expected to contain a reasonably clear view of
    the clothing item.
-   Visual similarity is based on the image representation learned by
    FashionCLIP and does not guarantee exact product identity.
-   Similarity scores represent visual embedding similarity and should
    not be interpreted as a human-verified percentage match.
-   Product stock and pricing are taken from the indexed catalogue
    metadata and may change on the source website.
-   The system assumes that the indexed catalogue contains sufficiently
    relevant visual examples for meaningful retrieval.

------------------------------------------------------------------------

## Trade-offs

### FashionCLIP

**Advantage:**\
Fashion-specific embeddings are better suited to clothing similarity
than generic image embeddings.

**Trade-off:**\
The model adds a relatively heavy inference dependency and increases
application startup time and resource requirements.

### Multi-view retrieval

**Advantage:**\
Searching multiple regions of the image can capture details that may be
underrepresented in a single full-image embedding.

**Trade-off:**\
Each query requires multiple embedding generations and vector searches,
increasing inference and retrieval time compared with a single-view
search.

### Candidate expansion and reranking

**Advantage:**\
Retrieving 20 candidates per view before reranking provides a larger
candidate pool and allows products to be evaluated across multiple
visual views.

**Trade-off:**\
The additional retrieval and ranking steps increase query latency.

### Qdrant

**Advantage:**\
Qdrant provides a dedicated vector-search layer and is more appropriate
for scalable similarity retrieval than performing a full in-memory
comparison for every query.

**Trade-off:**\
The application depends on an external vector database service and
therefore requires Qdrant configuration and network access.

### Groq + LangChain

**Advantage:**\
The combination provides a simple way to integrate an LLM response layer
with a structured search tool.

**Trade-off:**\
An external LLM API is required for natural-language responses.

### Top-5 retrieval

**Advantage:**\
Returning five candidates gives users multiple alternatives to compare
instead of relying on a single result.

**Trade-off:**\
Lower-ranked results may be less visually similar than the top-ranked
result.

------------------------------------------------------------------------

## Current Features

-   ✅ Fashion image upload
-   ✅ FashionCLIP image embeddings
-   ✅ Multi-view visual retrieval
-   ✅ Qdrant vector search
-   ✅ Candidate expansion and weighted reranking
-   ✅ Top-5 visual similarity retrieval
-   ✅ LangChain callable search tool
-   ✅ Groq LLM response generation
-   ✅ Similarity scores
-   ✅ Price and discount information
-   ✅ Stock availability
-   ✅ Product website links
-   ✅ Product images in the Streamlit results UI
-   ✅ Product-focused conversational follow-up
-   ✅ Testing with multiple saree images
-   ✅ Deployed Streamlit application

------------------------------------------------------------------------

## Deployment

TailorTalk is deployed using **Streamlit Community Cloud**.

The deployed application is available at:

``` text
https://tailortalk-demo.streamlit.app/
```

The deployment requires the following secrets to be configured:

``` text
QDRANT_URL
QDRANT_API_KEY
GROQ_API_KEY
```

The local `.env` file is not committed to the repository.

------------------------------------------------------------------------

## Assignment Alignment

TailorTalk implements the assignment's core visual-search pipeline:

``` text
Fashion Image
      ↓
Multi-View Image Processing
      ↓
FashionCLIP Embeddings
      ↓
Qdrant Vector Search
      ↓
Candidate Combination
      ↓
Weighted Reranking
      ↓
Top-5 Similar Products
      ↓
Groq LLM
      ↓
Streamlit Interface
```

The implementation specifically focuses on improving visual similarity
rather than relying on generic category matching.

### Final Submission

The project provides:

1.  A working deployed application
2.  A GitHub repository containing the source code
3.  Setup instructions
4.  Architecture and technology choices
5.  Visual retrieval quality improvements
6.  Assumptions and trade-offs
7.  Testing utilities and documentation

------------------------------------------------------------------------

## Conclusion

TailorTalk combines a fashion-specific vision model, vector similarity
search, multi-view retrieval, weighted reranking and an LLM-powered
shopping interface to provide visually similar saree recommendations
from a catalogue.

The main focus of the implementation is not simply retrieving products
from the same category, but improving the relevance of the returned
products by considering both **global appearance and localized visual
details**.
