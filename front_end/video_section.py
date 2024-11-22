import os
import streamlit as st
import cv2
import requests
import base64
import time
import tempfile
from PIL import Image, ImageSequence
api_url_video = st.secrets["API_URL_video"]
def display_video_section(video_path, word):
    ##placeholder to display the HIPPO video
#    word = "HIPPO"
    st.markdown(f'Do you want to learn how to sign the word {word}?')

    st.video(os.path.join(video_path, f'{word}.mp4'), start_time=0.10)

    lbl = f'Record your {word} sign video, or upload a video'
    #camera_input = st.camera_input(label=lbl, key=f"camera_3_{st.session_state.page_key['Game On!']}")

#    uploaded_file = st.file_uploader("Upload video", type=["jpg", "jpeg", "png"])
    uploaded_file = st.file_uploader("", type=["mp4"])

    #if camera_input:
        #st.markdown("Sending video to API for prediction")

    if uploaded_file:
        st.markdown("Sending video to API for prediction")




    
    gif_placeholder = st.empty()
    progress_bar = st.progress(0)
    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        # Open the video file using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return
        progress = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_base64 = []
        frames = []
        # Read all frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                st.info("Video file uploaded...relax and wait for prediction.")
                break
            # Resize the frame to reduce size
            frame = cv2.resize(frame, (320, 240))
            # Convert frame to JPEG format and encode it to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_base64)
            # Convert frame to RGB and store it for GIF creation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            # Update the progress bar
            progress += 1
            progress_bar.progress(progress / total_frames)
            time.sleep(0.02)
        # Create GIF from the frames
        gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=30)
        gif_placeholder.image(gif_path)
        # Send all frames to the API at once
        payload = {"frames": frames_base64}
        try:
            response = requests.post(api_url_video, json=payload)
            if response.status_code == 200:
                result = response.json()
                prediction_text = f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}%"
                # Check if GIF is included in the response
                if 'gif' in result:
                    gif_base64 = result['gif']
                    gif_bytes = base64.b64decode(gif_base64)
                    gif_placeholder.image(gif_bytes)#, format="GIF")
                st.success(prediction_text)
                return 'success'
            else:
                st.error(f"Error: Received status code {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
        # Release the video resource
        cap.release()
