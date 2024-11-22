import os
import streamlit as st

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
