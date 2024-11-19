import streamlit as st
import numpy as np
import time
from PIL import Image
import requests
import io
import os
import random
import base64
from io import BytesIO
from front_ASL_layout import display_image_columns, adjust_brightness_contrast

# API URL from secrets
api_url = st.secrets["API_URL"]

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home Page", "Game On!"])

# Sidebar with project information
st.sidebar.title("Project Information")
st.sidebar.write("""
**Project Name**: Sign Language Recognition

**Goal**: Develop a model to recognize sign language using computer vision and machine learning techniques.

**Course**: LeWagon #1705 (Diana, Robert, Jean-Michel, Gabriel & Boris).

**Technologies Used**:
- Python for scripting and data processing
- Mediapipe and OpenCV for image processing and hand landmark detection
- TensorFlow for machine learning model development
- Streamlit for the web interface
""")

# Display reference images for letter signs
IMAGE_FOLDER = "front_end/asl"
image_files = sorted(
    [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')],
    key=lambda x: os.path.basename(x).lower()
)

with st.sidebar:
    st.markdown("## Reference Images for Letter Signs")
    with st.expander("Click to open"):
        cols = st.columns(4)
        for i, img_path in enumerate(image_files):
            with cols[i % 4]:
                img = Image.open(img_path)
                st.image(img, use_column_width=True, caption=os.path.basename(img_path).split('.')[0].capitalize())

# Cache clearing logic
LAST_CLEAR_TIME = 0
CACHE_CLEAR_INTERVAL = 120  # Interval in seconds (clear cache every 120 seconds)

def should_clear_cache():
    global LAST_CLEAR_TIME
    current_time = time.time()
    if current_time - LAST_CLEAR_TIME > CACHE_CLEAR_INTERVAL:
        LAST_CLEAR_TIME = current_time
        return True
    return False

def base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img

def get_predictions_with_progress(uploaded_file):
    if should_clear_cache():
        st.cache_data.clear()

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Prepare image data for API
    if isinstance(uploaded_file, np.ndarray):
        image = Image.fromarray(uploaded_file)
    else:
        image = Image.open(uploaded_file)

    # Resize image while maintaining aspect ratio
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 512
        new_height = int((512 / original_width) * original_height)
    else:
        new_height = 512
        new_width = int((512 / original_height) * original_width)

    image = image.resize((new_width, new_height))

    # Convert the image to a bytes object to send to the API
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')  # Save the image in PNG format
    image_data.seek(0)

    # Send the image to the API
    url = f"{api_url}/upload"
    files = {'file': image_data}

    # Update progress bar
    progress_bar.progress(10)

    response = requests.post(url, files=files)

    # Simulate progress bar for API processing
    for i in range(20, 100, 20):  # Increment progress in steps of 20%
        time.sleep(0.1)  # Simulate delay
        progress_bar.progress(i)

    # Process the API response
    prediction = response.json()['message']
    confidence = float(response.json()['confidence'][:5])  # Ensure confidence is a float

    processed_image = base64_image(response.json()["image"])
    hand_region = base64_image(response.json()["hand"])

    # Update progress bar to 100%
    progress_bar.progress(100)
    progress_bar.empty()

    return prediction, confidence, processed_image, hand_region

def calculate_score(predicted_letter, required_letter, confidence):
    score = min(round(confidence * 10), 10)  #0-10
    if predicted_letter != required_letter:
        return "Wrong Letter", "red", 0  # Wrong letter gets a score of 0
    elif score <= 4:
        return f"{score}/10", "red", score
    elif score <= 8:
        return f"{score}/10", "orange", score
    else:
        return f"{score}/10", "green", score

# Home Page functionality
if page == "Home Page":
    st.title("Show hands!")
    st.write("Take a picture with the computer camera, or upload a file.")

    # Camera and image input layout
    image_col, camera_col = st.columns([2, 10])

    IMAGE_FOLDER = "front_end/asl"
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]

    if "random_images" not in st.session_state:
        st.session_state.random_images = random.sample(image_files, 3)

    with image_col:
        # Display random image
        for img_path in st.session_state.random_images:
            img = Image.open(img_path)
            st.image(img, width=80)

        # Refresh random images
        if st.button("Refresh"):
            st.session_state.random_images = random.sample(image_files, 3)
    bright= st.slider("Select brightness",10,60,step=10,value = 30)
    contrast = st.slider("Select constrast",0.5,1.5,step = 0.25,value = 1.0)
    with camera_col:
        # Camera input
        camera_image = st.camera_input("Take a picture")

    # THIS IS THE WAY IT WAS BEFORE BO CHANGED IT #####################################################################################
    # hand_region = None
    # if camera_image:
    #     try:
    #         img_c = adjust_brightness_contrast(camera_image,brightness = bright,contrast =contrast)
    #         st.info("Processing...")
    #         prediction, confidence, processed_image, hand_region = get_predictions_with_progress(img_c)
    #     except Exception:
    #         st.write("Try again. Here is what we see:")
    #         st.image(img_c)

    # # File uploader for image input
    # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # if uploaded_file:

    #     try:
    #         img_u = adjust_brightness_contrast(uploaded_file,brightness = bright,contrast =contrast)
    #         st.info("Processing...")
    #         prediction, confidence, processed_image, hand_region = get_predictions_with_progress(img_u)
    #     except Exception:
    #         st.write("Try again. Here is what we see:")
    #         st.image(img_u)

        hand_region = None
        if camera_image:
            try:
                img_c = adjust_brightness_contrast(camera_image, brightness=bright, contrast=contrast)

                with st.spinner('Processing...'):
                    prediction, confidence, processed_image, hand_region = get_predictions_with_progress(img_c)
            except Exception:
                st.write("Try again. Here is what we see:")
                st.image(img_c)

            try:
                img_u = adjust_brightness_contrast(uploaded_file,brightness = bright,contrast =contrast)
                with st.spinner('Processing...'):
                    prediction, confidence, processed_image, hand_region = get_predictions_with_progress(img_u)
            except Exception:
                st.write("Try again. Here is what we see:")
                st.image(img_u)

    # Display processed results
    if hand_region:
        display_image_columns(processed_image, hand_region, (prediction, confidence))
    else:
        st.write("No hand detected in the image.")



# Game On! functionality
elif page == "Game On!":
    st.title("Game On!")

    # Initialize session state variables
    if "current_word" not in st.session_state:
        st.session_state.current_word = None
    if "current_letter_index" not in st.session_state:
        st.session_state.current_letter_index = 0
    if "letter_scores" not in st.session_state:
        # To store scores for each letter
        st.session_state.letter_scores = []
    if "game_files" not in st.session_state:
        st.session_state.game_files = [f for f in os.listdir("front_end/game_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
    if "selected_game_image" not in st.session_state:
        st.session_state.selected_game_image = random.choice(st.session_state.game_files)
    if "camera_input_key" not in st.session_state:
        # Initialize key for camera input
        st.session_state.camera_input_key = 0

    # Refresh word and image logic
    def refresh_game_image():
        st.session_state.selected_game_image = random.choice(st.session_state.game_files)
        st.session_state.current_word = os.path.splitext(st.session_state.selected_game_image)[0].upper()
        st.session_state.current_letter_index = 0
        st.session_state.letter_scores = [None] * len(st.session_state.current_word)  # Reset scores for each letter

    # Initialize current word and image
    if not st.session_state.current_word:
        refresh_game_image()

    game_image_path = os.path.join("front_end/game_images", st.session_state.selected_game_image)
    game_word = st.session_state.current_word
    current_letter = game_word[st.session_state.current_letter_index]

    # Layout for the Game On! page
    col_left, col_right = st.columns([2, 3])

    # Top-right refresh button
    with col_right:
        if st.button("Refresh Image"):
            refresh_game_image()

    # Left side: Display image and word
    with col_left:
        st.image(game_image_path, caption=f"Sign the word: {game_word}", use_column_width=True)
        st.markdown(f"### Current Letter: {current_letter}")
        st.markdown("### Letters Progress:")
        for idx, letter in enumerate(game_word):
            if idx < st.session_state.current_letter_index:
                # Completed letters: Display their icons based on score
                score = st.session_state.letter_scores[idx]
                if score is None or score <= 4:
                    st.markdown(f"❌ {letter} - And your score is {score}/10")
                elif score <= 8:
                    st.markdown(f"⚠️ {letter} - And your score is {score}/10")
                else:
                    st.markdown(f"✅ {letter} - And your score is {score}/10")
            elif idx == st.session_state.current_letter_index:
                # Highlight the current letter
                st.markdown(f"👉 {letter}")
            else:
                # Upcoming letters remain blank
                st.markdown(f"⬜ {letter}")

    # Right side: Camera input and buttons
    with col_right:
        st.markdown("### Sign Input:")
        camera_input = st.camera_input("Take a picture", key=st.session_state.camera_input_key)
        predicted_letter_placeholder = st.empty()  # Placeholder for predicted letter

        # Buttons: Try Again and Move On
        col_try_again, col_move_on = st.columns(2)
        try_again = col_try_again.button("Try Again")
        move_on = col_move_on.button("Move On")

        if camera_input:
            try:
                # Process the camera input and get the prediction
                results = get_predictions_with_progress(camera_input)
                prediction, confidence, _, _ = results
                predicted_letter = prediction.strip().split()[-1].upper()

                # Display the predicted letter
                predicted_letter_placeholder.markdown(
                    f"<div style='font-size:20px; font-weight:bold;'>Predicted Letter: <span style='color:blue;'>{predicted_letter}</span></div>",
                    unsafe_allow_html=True
                )

                # Check the current letter and calculate score
                if predicted_letter == current_letter:
                    score_text, color, score = calculate_score(predicted_letter, current_letter, confidence)
                    st.session_state.letter_scores[st.session_state.current_letter_index] = score

                    st.markdown(
                        f"<div style='color:{color}; font-size:18px;'>Prediction Correct! {current_letter}: {score_text}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='color:red; font-size:18px;'>Wrong Letter! Expected: {current_letter}</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.letter_scores[st.session_state.current_letter_index] = 0

            except Exception as e:
                st.error(f"Error processing input: {e}")

        # Button logic for Try Again and Move On
        if try_again or move_on:
            # Clear the camera input by incrementing the camera_input_key
            st.session_state.camera_input_key += 1
            camera_input = None  # Clear the camera input placeholder

        if move_on:
            # Save the last score and move to the next letter
            st.session_state.letter_scores[st.session_state.current_letter_index] = (
                st.session_state.letter_scores[st.session_state.current_letter_index] or 0
            )

            if st.session_state.current_letter_index < len(game_word) - 1:
                st.session_state.current_letter_index += 1
                predicted_letter_placeholder.empty()
            else:
                st.write("You have completed the word!")
                refresh_game_image()


# Function to display the API URL (for debugging)
def display_url():
    print(api_url)

# Run the script
if __name__ == "__main__":
    display_url()
