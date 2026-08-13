from search_tool import search_similar_products


IMAGE_PATH = "data/images/000691.jpg"


print("\nTesting LangChain search tool...\n")

results = search_similar_products.invoke({
    "image_path": IMAGE_PATH
})


print("=" * 60)
print("TOOL RESULTS")
print("=" * 60)

for i, product in enumerate(results, 1):

    print(f"\n#{i}")

    print("Score:", product["score"])
    print("Name:", product["name"])
    print("SKU:", product["sku"])
    print("Stock:", product["stock"])
    print("Retail Price:", product["retail_price"])
    print("Discounted Price:", product["discounted_price"])
    print("Website:", product["website_link"])