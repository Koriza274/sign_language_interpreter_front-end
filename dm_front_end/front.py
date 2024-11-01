API_URL = "https://dmapi-564221756825.europe-west1.run.app"

import streamlit as st
from PIL import Image
import requests
import io

def get_predictions(image_data):
    url = f"{API_URL}/upload"
    files = {'file': image_data}
    response = requests.post(url, files=files)
    return response.json()['message']


st.title("Image Upload for Prediction")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)


    # Prepare image data for API
    image_data = io.BytesIO()
    image.save(image_data,format = 'PNG')  # Save the image in the desired format
    image_data.seek(0)  # Rewind the buffer

    # Make a prediction
    if st.button("Predict"):
        predictions = get_predictions(image_data)
        st.write("Predictions:")
        st.write(predictions)
