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

    st.markdown("---")

    st.markdown("## 🤖 TailorTalk")

    st.markdown(
        st.session_state.search_result["response"]
    )