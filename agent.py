import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage

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
# TOOLS
# --------------------------------------------------

tools = [
    search_similar_products
]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are TailorTalk, an AI fashion shopping assistant.

Your job is to help users discover visually similar
fashion products from the TailorTalk catalogue.

When the user provides an image and asks for similar
products, use the search_similar_products tool.

The tool accepts an image_path and searches the product
catalogue using FashionCLIP embeddings and Qdrant.

After receiving search results:

- Recommend the most relevant products.
- Mention similarity scores.
- Mention prices.
- Mention stock availability.
- Include the product website when available.
- Keep the response concise and useful.

Never claim that you searched the catalogue unless
the search tool was actually executed.
"""


# --------------------------------------------------
# AGENT FUNCTION
# --------------------------------------------------

def search_products(image_path):

    messages = [
        (
            "system",
            SYSTEM_PROMPT
        ),
        (
            "human",
            f"""
            Find fashion products visually similar to this image:

            {image_path}
            """
        )
    ]

    # --------------------------------------------------
    # STEP 1: ASK LLM
    # --------------------------------------------------

    response = llm_with_tools.invoke(messages)

    print("\nLLM tool call:")

    if not response.tool_calls:
        print("No tool call generated.")
        return

    # --------------------------------------------------
    # STEP 2: EXECUTE TOOL
    # --------------------------------------------------

    tool_call = response.tool_calls[0]

    print("Tool:", tool_call["name"])
    print("Arguments:", tool_call["args"])

    tool_result = search_similar_products.invoke(
        tool_call["args"]
    )

    print("\nTool executed successfully.")

    # --------------------------------------------------
    # STEP 3: ADD AI MESSAGE + TOOL RESULT
    # --------------------------------------------------

    messages.append(response)

    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # --------------------------------------------------
    # STEP 4: FINAL LLM RESPONSE
    # --------------------------------------------------

    final_response = llm.invoke(messages)

    print("\n" + "=" * 60)
    print("TAILORTALK RESPONSE")
    print("=" * 60)

    print(final_response.content)
# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting TailorTalk agent...\n")

    search_products(
        "data/images/000691.jpg"
    )