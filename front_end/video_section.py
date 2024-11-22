import os
import streamlit as st

def display_video_section(video_path, word):
    """
    Displays the video section for the given word, allowing the user to learn how to sign it.
    """
    st.title(f"Learn to sign the word: {word}")
    st.video(os.path.join(video_path, f'{word}.mp4'), start_time=0.10)

    lbl = f'Record your {word} sign video, or upload a video'
    camera_input = st.camera_input(label=lbl, key=f"camera_3_{st.session_state.page_key['Game On!']}")

    uploaded_file = st.file_uploader("Upload your sign video", type=["mp4"])

    if camera_input:
        st.markdown("Sending video to API for prediction...")

    if uploaded_file:
        st.markdown("Sending uploaded video to API for prediction...")

