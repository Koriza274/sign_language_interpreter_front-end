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
from video_section import display_video_section

# API URL from secrets
api_url = st.secrets["API_URL"]
# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home Page", "Game On!"])

# Sidebar with project information
st.sidebar.title("Learn American Sign Language")
st.sidebar.write("""
**Goal**: This project is designed to help users learn the American Sign Language (ASL) alphabet by recognizing and mimicking gestures for each letter. It also features an interactive game mode where users can practice spelling words using ASL.

**Background**: This project was developed as part of the LeWagon #1705 course by the following contributors: [Jean-Michel](https://github.com/JMLejeune-evolvi), [Diana](https://github.com/Koriza274/), [Robert](https://github.com/ropath), [Gabriel](https://github.com/gabrielrehder) & [Boris](https://github.com/just1984).

**Technologies Used**:
- **Python**: For scripting and data processing
- **Mediapipe & OpenCV**: For image processing and hand landmark detection
- **TensorFlow**: For building and training the recognition model
- **Streamlit**: For creating an interactive web interface
""")

#IMAGE_FOLDER = os.path.join(os.getcwd(), 'asl')
#GAME_IMAGES = os.path.join(os.getcwd(), 'game_images')

IMAGE_FOLDER = "front_end/asl"
GAME_IMAGES = "front_end/game_images"
VIDEO_FOLDER = "front_end/videos"

if "image_files" not in st.session_state:
    # Display reference images for letter signs
    image_files = sorted(
        [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')],
        key=lambda x: os.path.basename(x).lower()
    )
    st.session_state.image_files = image_files

if "game_images" not in st.session_state:
    # Retreive game images
    st.session_state.game_images = st.session_state.game_files = [f for f in os.listdir(GAME_IMAGES) if f.endswith(('.png', '.jpg', '.jpeg'))]

#Initialize camera session_state variables
if "camera_input" not in st.session_state:
    st.session_state.camera_input = None  # To store the image
if "camera_input_key" not in st.session_state:
    # Initialize key for camera input
    st.session_state.camera_input_key = 0
if "clear_requested" not in st.session_state:
    st.session_state.clear_requested = False
if "camera_cleared" not in st.session_state:
    st.session_state.camera_cleared = False  # Flag to manage camera reset
if "java_clear_script" not in st.session_state:
    st.session_state.java_clear_script = False
# Initialize session state for page-specific keys
if "page_key" not in st.session_state:
    st.session_state.page_key = {"Home Page": 0, "Game On!": 0}  # Separate keys for pages
if "user_gives_up" not in st.session_state:
    st.session_state.user_gives_up = False
if "challenge_completed" not in st.session_state:
    st.session_state.challenge_completed = False

def reset_camera(page_name):
    st.session_state.page_key[page_name] += 1  # Increment key for the current page
    st.session_state[f"{page_name}_camera_input"] = None  # Clear the captured image


def clear_camera_picture():
    st.session_state.camera_input_key += 1  # Increment key to reset component
    st.session_state.camera_input = None  # Clear stored image

with st.sidebar:
    st.markdown("## Reference Images for Letter Signs")
    with st.expander("Click to open"):
        cols = st.columns(4)
        for i, img_path in enumerate(st.session_state.image_files):
            with cols[i % 4]:
                img = Image.open(img_path)
                st.image(img, use_column_width=True, caption=os.path.basename(img_path).split('.')[0].capitalize())



# Cache clearing logic
LAST_CLEAR_TIME = 0
CACHE_CLEAR_INTERVAL = 240  # Interval in seconds (clear cache every 240 seconds)

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

    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_data.seek(0)

    # Send the image to the API
    url = f"{api_url}/upload"
    files = {'file': image_data}

    # Update progress bar
    progress_bar.progress(10)

    response = requests.post(url, files=files)

    for i in range(20, 100, 20):
        time.sleep(0.1)
        progress_bar.progress(i)

    # Process the API response
    prediction = response.json()['message']
    confidence = float(response.json()['confidence'][:5])

    processed_image = base64_image(response.json()["image"])
    hand_region = base64_image(response.json()["hand"])

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
    st.title("Show hands and learn how to sign!")
    st.write("Take a picture with the computer camera, or upload a file and try to mimic the signs on the left.")

    # Camera and image input layout
    image_col, camera_col = st.columns([2, 10])

#    IMAGE_FOLDER = "asl"
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png')]

    if "random_images" not in st.session_state:
        st.session_state.random_images = random.sample(image_files, 3)

    with image_col:
        for img_path in st.session_state.random_images:
            img = Image.open(img_path)
            st.image(img, width=80)

        if st.button("Refresh"):
            st.session_state.random_images = random.sample(image_files, 3)

    st.write("If your image is too dark or bright, you can adjust it here using these sliders.")
    col1, col2 = st.columns(2)

    with col1:
        bright = st.slider("Select brightness", 10, 60, step=10, value=30)

    with col2:
        contrast = st.slider("Select contrast", 0.5, 1.5, step=0.25, value=1.0)

    with camera_col:
        # Camera input
        camera_image = st.camera_input("Take a picture", key=f"camera_1_{st.session_state.page_key['Home Page']}")

    hand_region = None
    if camera_image:
        try:
            img_c = adjust_brightness_contrast(camera_image, brightness=bright, contrast=contrast)
            with st.spinner("Processing..."):
                prediction, confidence, processed_image, hand_region = get_predictions_with_progress(img_c)
        except Exception:
            st.write("Try again. Here is what we see:")
            st.image(img_c)

    # THE FOLLOWING DISPLAYS THE UPLOAD AN IMAGE SECTION ###########################################################
    # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # if uploaded_file:
    #     try:
    #         img_u = adjust_brightness_contrast(uploaded_file, brightness=bright, contrast=contrast)
    #         with st.spinner("Processing..."):
    #             prediction, confidence, processed_image, hand_region = get_predictions_with_progress(img_u)
    #     except Exception:
    #         st.write("Try again. Here is what we see:")
    #         st.image(img_u)

    # Display processed results
    if hand_region:
        display_image_columns(processed_image, hand_region, (prediction, confidence))
    else:
        st.write("No hand detected in the image.")



# Game On! functionality
elif page == "Game On!":
    st.title("Game On!")

    reset_camera("Home Page")

    # Initialize session state variables
    if "current_word" not in st.session_state:
        st.session_state.current_word = None
    if "current_letter_index" not in st.session_state:
        st.session_state.current_letter_index = 0
    if "letter_scores" not in st.session_state:
        # To store scores for each letter
        st.session_state.letter_scores = []
#    if "game_files" not in st.session_state:
#        st.session_state.game_files = [f for f in os.listdir("game_images") if f.endswith(('.png', '.jpg', '.jpeg'))]
    if "selected_game_image" not in st.session_state:
        st.session_state.selected_game_image = random.choice(st.session_state.game_files)

    # Refresh word and image logic
    def refresh_game_image():
        previous_image = st.session_state.selected_game_image
        new_image = random.choice(st.session_state.game_files)

        # Optional: Stelle sicher, dass ein neues Bild ausgewählt wird
        while new_image == previous_image and len(st.session_state.game_files) > 1:
            new_image = random.choice(st.session_state.game_files)

        st.session_state.selected_game_image = new_image
        st.session_state.current_word = os.path.splitext(st.session_state.selected_game_image)[0].upper()
        st.session_state.current_letter_index = 0
        st.session_state.letter_scores = [None] * len(st.session_state.current_word)
        st.session_state.user_gives_up = False
        st.session_state.challenge_completed = False

        if 'predicted_letter_placeholder' in st.session_state:
            st.session_state.predicted_letter_placeholder = st.empty()


    # Initialize current word and image
    if not st.session_state.current_word:
        refresh_game_image()

    # Layout for the Game On! page
#    col_left, col_right = st.columns([2, 3])
    col_left, col_right = st.columns(2)

    # Right side: Camera input and buttons
    with col_right:
        st.markdown("### ⭐ Your turn! ⭐")

        lbl = "With your computer camera, take pictures of you signing each letter for the word challenge. Retry as many times as you want, clearing and taking pictures until you succeed, or move on to the next letter!"
        camera_input = st.camera_input(label=lbl, key=f"camera_2_{st.session_state.page_key['Game On!']}")
        predicted_letter_placeholder = st.empty()  # Placeholder for predicted letter
        if 'predicted_letter_placeholder' not in st.session_state:
            st.session_state.predicted_letter_placeholder = predicted_letter_placeholder

        st.markdown("""
                    <style>
                    div.stButton > button {width: 100%;
                    }
                    </style>
                    """, unsafe_allow_html=True)

        col_move_on, col_change_animal = st.columns(2)
        with col_move_on:
            move_on = col_move_on.button("Skip letter !")

        with col_change_animal:
            change_animal = col_change_animal.button("Change animal")

        # Refresh random images
        if change_animal:

            st.session_state.random_images = random.sample(st.session_state.game_images, 3)
            refresh_game_image()

        else:

            #only predict when the user is taking a picture, not when they're asking to move on
    #        if camera_input:
            if camera_input and not move_on:

                try:

                    st.session_state.camera_input = camera_input
                    st.query_params.clear()

                    # Process the camera input and get the prediction
                    results = get_predictions_with_progress(st.session_state.camera_input)
                    prediction, confidence, _, _ = results
                    predicted_letter = prediction.strip().split()[-1].upper()

                    # Display the predicted letter
                    predicted_letter_placeholder.markdown(
                        f"<div style='font-size:20px; font-weight:bold;'>Predicted Letter: <span style='color:blue;'>{predicted_letter}</span></div>",
                        unsafe_allow_html=True
                    )

                    # Check the current letter and calculate score
    #                if predicted_letter == current_letter:
                    if predicted_letter == st.session_state.current_word[st.session_state.current_letter_index]:
    #                    score_text, color, score = calculate_score(predicted_letter, current_letter, confidence)
                        score_text, color, score = calculate_score(predicted_letter, st.session_state.current_word[st.session_state.current_letter_index], confidence)
                        st.session_state.letter_scores[st.session_state.current_letter_index] = score

                        st.markdown(
    #                        f"<div style='color:{color}; font-size:18px;'>Prediction Correct! {current_letter}: {score_text}</div>",
                            f"<div style='color:{color}; font-size:18px;'>Prediction Correct! {st.session_state.current_word[st.session_state.current_letter_index]}: {score_text}</div>",
                            unsafe_allow_html=True,
                        )

                        st.session_state.current_letter_index +=1

    #                    if st.session_state.current_letter_index == len(game_word):
                        if st.session_state.current_letter_index == len(st.session_state.current_word):
                            st.write("You have completed the word!")
                            st.session_state.challenge_completed = True


                    else:
                        st.markdown(
    #                        f"<div style='color:red; font-size:18px;'>Wrong Letter! Expected: {current_letter}</div>",
                            f"<div style='color:red; font-size:18px;'>Wrong Letter! Expected: {st.session_state.current_word[st.session_state.current_letter_index]}</div>",
                            unsafe_allow_html=True,
                        )
                        st.session_state.letter_scores[st.session_state.current_letter_index] = 0

                except Exception as e:
                    st.error(f"Error processing input: {e}")

            # Button logic for Try Again and Move On
    #        if try_again or move_on:
            if move_on:
                reset_camera('Game On!')
    #            camera_input = None  # Clear the camera input placeholder

            if move_on:
                # Save the last score and move to the next letter
                st.session_state.letter_scores[st.session_state.current_letter_index] = (
                    st.session_state.letter_scores[st.session_state.current_letter_index] or 0
                )

                st.session_state.current_letter_index += 1

    #            if st.session_state.current_letter_index < len(game_word) - 1:
                if st.session_state.current_letter_index <= len(st.session_state.current_word) - 1:
    #                st.session_state.current_letter_index += 1
                    predicted_letter_placeholder.empty()
                else:
                    st.session_state.user_gives_up = True
                    st.session_state.challenge_completed = True

    # Left side: Display image and word
    with col_left:
#        st.image(game_image_path, caption=f"Sign the word: {game_word}", use_container_width=True)
        st.image(os.path.join(GAME_IMAGES, st.session_state.selected_game_image), use_column_width=True)



        if st.session_state.current_letter_index == len(st.session_state.current_word):
            current_letter = "Done!"
        else:
            current_letter = st.session_state.current_word[st.session_state.current_letter_index]
        st.markdown(f"### Current Letter: {current_letter}")
        st.markdown("### Letters score:")
#        for idx, letter in enumerate(game_word):
        for idx, letter in enumerate(st.session_state.current_word):
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

        if  st.session_state.user_gives_up:
#            st.markdown(f"You are giving up on {game_word}!")
            st.markdown(f"You are giving up on {st.session_state.current_word}!")

        if st.session_state.challenge_completed:
            with col_right:
                display_video_section(VIDEO_FOLDER, st.session_state.current_word)
            current_word = st.session_state.current_word
            ##clean up session_state for challenge
            st.session_state.current_word = None
            st.session_state.current_letter_index = 0
            st.session_state.letter_scores = []
            st.session_state.selected_game_image = random.choice(st.session_state.game_files)
            predicted_letter_placeholder.empty()
            st.session_state.user_gives_up = False
            st.session_state.challenge_completed = False


# Function to display the API URL (for debugging)
def display_url():
    print(api_url)

# Run the script
if __name__ == "__main__":
    display_url()
