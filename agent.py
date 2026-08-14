import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from search_tool import search_similar_products


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )


# --------------------------------------------------
# MODEL
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are TailorTalk, an AI fashion shopping assistant.

You help users discover visually similar fashion
products from the TailorTalk catalogue.

The visual search itself is performed by the
search_similar_products tool using FashionCLIP
embeddings and Qdrant.

When given search results:

- Recommend the most relevant products.
- Mention similarity scores when useful.
- Mention prices.
- Mention stock availability.
- Include product website links when available.
- Keep responses concise and useful.
- Never invent products, prices, stock, scores or links.
"""


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def search_products(image_path):

    # --------------------------------------------------
    # STEP 1: DIRECTLY EXECUTE VISUAL SEARCH
    # --------------------------------------------------

    tool_result = search_similar_products.invoke(
        {
            "image_path": image_path
        }
    )

    # --------------------------------------------------
    # STEP 2: ASK LLM TO FORMAT THE RESULTS
    # --------------------------------------------------

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=f"""
A user uploaded a fashion image and wants visually
similar products.

Here are the results returned by the visual search
system:

{tool_result}

Create a concise shopping response for the user.

Do not perform another search.
Do not invent information.
"""
        )
    ]

    final_response = llm.invoke(messages)

    return {
        "response": final_response.content,
        "products": tool_result
    }


# --------------------------------------------------
# CLI TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting TailorTalk agent...\n")

    result = search_products(
        "data/images/000691.jpg"
    )

    print("\n" + "=" * 60)
    print("TAILORTALK RESPONSE")
    print("=" * 60)

    print(result["response"])

    # --------------------------------------------------
# CHAT ABOUT SEARCH RESULTS
# --------------------------------------------------

def chat_about_products(
    question,
    products,
    chat_history=None
):

    product_context = ""

    for index, product in enumerate(
        products[:5],
        start=1
    ):

        stock = max(
            0,
            int(product.get("stock") or 0)
        )

        product_context += f"""
Product #{index}

Name: {product.get("name")}
SKU: {product.get("sku")}
Similarity: {product.get("score")}
Stock: {stock}
Retail Price: {product.get("retail_price")}
Discounted Price: {product.get("discounted_price")}
Product URL: {product.get("website_link")}

"""


    messages = [
        SystemMessage(
            content="""
You are TailorTalk, an AI fashion shopping assistant.

The user has already performed a visual search.

You must answer questions ONLY using the products
returned by the visual search.

Do not invent:
- products
- prices
- stock
- similarity scores
- URLs
- product features

If the information is not available in the search
results, clearly say that it is not available.

Be concise and helpful.

If the user asks:
- cheapest → compare discounted prices
- most similar → compare similarity scores
- available/in stock → use stock values
- product link → provide the corresponding URL
- best option → make a reasonable recommendation
  based only on similarity, price and availability.
"""
        )
    ]


    # --------------------------------------------------
    # ADD PREVIOUS CHAT
    # --------------------------------------------------

    if chat_history:

        for message in chat_history[-6:]:

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
                if message["role"] == "user"
                else SystemMessage(
                    content=message["content"]
                )
            )


    # --------------------------------------------------
    # CURRENT QUESTION + PRODUCT DATA
    # --------------------------------------------------

    messages.append(
        HumanMessage(
            content=f"""
Here are the current visual search results:

{product_context}

User question:

{question}

Answer the user's question using only these results.
"""
        )
    )


    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    response = llm.invoke(
        messages
    )

    return response.content