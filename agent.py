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