import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def display_image_columns(processed_image, hand_region, prediction_and_confidence):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(processed_image, caption="Original", use_container_width=True)
    with col2:
        st.image(hand_region, caption="Hand region")
    with col3:
        prediction, confidence = prediction_and_confidence
        st.markdown(
            f"<p style='font-size:22px; font-weight:bold;'>{prediction}</p>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence: {confidence}%")


def adjust_brightness_contrast(image, brightness=40, contrast=1.0):
    """
    Adjust the brightness and contrast of an image.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(BytesIO(image.getvalue()))

    img = np.array(image).astype(np.float32)

    img = img * contrast + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)

    adjusted_image = Image.fromarray(img)

    return adjusted_image
