import os
import tempfile

import streamlit as st

from agent import search_products, chat_about_products


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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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

            # Store search result
            st.session_state.search_result = result

            # Reset chat for new image
            st.session_state.chat_history = []

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


# ==================================================
# DISPLAY RESULTS
# ==================================================

if st.session_state.search_result:

    result = st.session_state.search_result

    products = result.get(
        "products",
        []
    )

    # --------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------

    st.markdown("---")

    st.markdown("## 🤖 TailorTalk")

    st.markdown(
        result["response"]
    )


    # ==================================================
    # TOP 5 PRODUCTS
    # ==================================================

    if products:

        st.markdown("---")

        st.markdown(
            "## 🛍️ Top 5 Visual Matches"
        )

        cols = st.columns(5)

        for index, product in enumerate(
            products[:5]
        ):

            with cols[index]:

                # --------------------------------------------------
                # PRODUCT IMAGE
                # --------------------------------------------------

                image_url = product.get(
                    "image_url"
                )

                if image_url:

                    st.image(
                        image_url,
                        use_container_width=True
                    )

                # --------------------------------------------------
                # PRODUCT NAME
                # --------------------------------------------------

                st.markdown(
                    f"**#{index + 1} "
                    f"{product.get('name', 'Unknown Product')}**"
                )

                # --------------------------------------------------
                # SIMILARITY
                # --------------------------------------------------

                score = product.get(
                    "score"
                )

                if score is not None:

                    st.write(
                        f"🎯 Similarity: "
                        f"**{score:.4f}**"
                    )

                # --------------------------------------------------
                # PRICE
                # --------------------------------------------------

                discounted = product.get(
                    "discounted_price"
                )

                retail = product.get(
                    "retail_price"
                )

                if discounted is not None:

                    st.write(
                        f"💰 **₹{discounted:,.0f}**"
                    )

                    if (
                        retail is not None
                        and retail != discounted
                    ):

                        st.caption(
                            f"Retail: "
                            f"₹{retail:,.0f}"
                        )

                elif retail is not None:

                    st.write(
                        f"💰 **₹{retail:,.0f}**"
                    )

                # --------------------------------------------------
                # STOCK
                # --------------------------------------------------

                stock = product.get(
                    "stock",
                    0
                )

                stock = max(
                    0,
                    int(stock or 0)
                )

                if stock > 0:

                    st.success(
                        f"✓ {stock} in stock"
                    )

                else:

                    st.error(
                        "✕ Out of stock"
                    )

                # --------------------------------------------------
                # PRODUCT URL
                # --------------------------------------------------

                website = product.get(
                    "website_link"
                )

                if website:

                    st.link_button(
                        "View Product ↗",
                        website,
                        use_container_width=True
                    )


    # ==================================================
    # CHAT UI
    # ==================================================

    st.markdown("---")

    st.markdown(
        "## 💬 Ask TailorTalk"
    )

    st.caption(
        "Ask anything about the products above."
    )


    # --------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------

    user_question = st.chat_input(
        "e.g. Which one is the cheapest?"
    )


    if user_question:

        # --------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_question
            )


        # --------------------------------------------------
        # AI RESPONSE
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    response = chat_about_products(
                        user_question,
                        products,
                        st.session_state.chat_history
                    )

                except Exception as e:

                    response = (
                        f"Sorry, something went wrong: {e}"
                    )

            st.markdown(
                response
            )


        # --------------------------------------------------
        # SAVE AI RESPONSE
        # --------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )