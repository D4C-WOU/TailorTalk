import os
import tempfile

import streamlit as st

from agent import search_products


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="TailorTalk",
    page_icon="👗",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "search_result" not in st.session_state:
    st.session_state.search_result = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("👗 TailorTalk")

st.subheader(
    "AI-powered visual fashion search"
)

st.write(
    "Upload a fashion image and discover visually "
    "similar products from our catalogue."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a fashion/product image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file:

    st.image(
        uploaded_file,
        caption="Your uploaded image",
        width=400
    )

    # --------------------------------------------------
    # SEARCH BUTTON
    # --------------------------------------------------

    if st.button(
        "🔍 Find Similar Products",
        type="primary"
    ):

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        image_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                image_path = temp_file.name

            # --------------------------------------------------
            # SEARCH
            # --------------------------------------------------

            with st.spinner(
                "Finding visually similar products..."
            ):

                result = search_products(
                    image_path
                )

            # Store result instead of immediately rendering it
            st.session_state.search_result = result

        except Exception as e:

            st.session_state.search_result = None

            st.error(
                f"Search failed: {e}"
            )

        finally:

            if image_path:

                try:
                    os.remove(image_path)

                except OSError:
                    pass


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

if st.session_state.search_result:

    result = st.session_state.search_result

    st.markdown("---")

    # --------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------

    st.markdown("## 🤖 TailorTalk")

    st.markdown(
        result["response"]
    )

    # --------------------------------------------------
    # TOP 5 VISUAL MATCHES
    # --------------------------------------------------

    products = result.get("products", [])

    if products:

        st.markdown("---")
        st.markdown("## 🛍️ Top 5 Visual Matches")

        cols = st.columns(5)

        for index, product in enumerate(products[:5]):

            with cols[index]:

                # Product image
                image_url = product.get("image_url")

                if image_url:
                    st.image(
                        image_url,
                        use_container_width=True
                    )

                # Product name
                st.markdown(
                    f"**#{index + 1} {product.get('name', 'Unknown Product')}**"
                )

                # Similarity
                score = product.get("score")

                if score is not None:
                    st.write(
                        f"🎯 Similarity: **{score:.4f}**"
                    )

                # Price
                discounted = product.get(
                    "discounted_price"
                )

                retail = product.get(
                    "retail_price"
                )

                if discounted is not None:
                    st.write(
                        f"💰 ₹{discounted:,.0f}"
                    )

                    if retail is not None:
                        st.caption(
                            f"Retail: ₹{retail:,.0f}"
                        )

                elif retail is not None:
                    st.write(
                        f"💰 ₹{retail:,.0f}"
                    )

                # Stock
                stock = product.get("stock")

                if stock is not None:

                    if stock > 0:
                        st.success(
                            f"✓ {int(stock)} in stock"
                        )
                    else:
                        st.error(
                            "✕ Out of stock"
                        )

                # Website
                website = product.get(
                    "website_link"
                )

                if website:
                    st.link_button(
                        "View Product",
                        website,
                        use_container_width=True
                    )