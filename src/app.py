import streamlit as st
import cv2
import threading
import time
import numpy as np
from PIL import Image
from pipeline import MunsitPipeline
from visual_overlay import render_overlay

st.set_page_config(
    page_title="Munsit",
    page_icon="👁",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0a0f1a; }
    .block-container { padding-top: 2rem; }
    h1 { color: #00c9ff; }
    .stButton > button {
        background: linear-gradient(135deg, #00c9ff, #7b5ea7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return MunsitPipeline()

def main():
    st.title("Munsit — Real-Time Arabic Accessibility")
    st.caption("Speech · Sign Language · Environmental Audio")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Controls")
        camera_index = st.selectbox("Camera", [0, 1, 2], index=1)
        enable_speech = st.toggle("Speech Recognition", value=True)
        enable_sign = st.toggle("Sign Detection", value=True)
        enable_audio = st.toggle("Audio Events", value=True)
        start = st.button("Start", use_container_width=True)
        stop = st.button("Stop", use_container_width=True)

        st.divider()
        st.subheader("Live Output")
        text_box = st.empty()
        sign_box = st.empty()
        audio_box = st.empty()

    with col1:
        st.subheader("Camera Feed")
        frame_placeholder = st.empty()

    if start:
        pipeline = load_pipeline()
        cap = cv2.VideoCapture(camera_index)

        def speech_loop():
            while st.session_state.get("running", False):
                if enable_speech:
                    pipeline.transcribe_audio(duration=5)

        def audio_loop():
            while st.session_state.get("running", False):
                if enable_audio:
                    pipeline.classify_audio_event(duration=4)

        st.session_state["running"] = True

        t1 = threading.Thread(target=speech_loop, daemon=True)
        t2 = threading.Thread(target=audio_loop, daemon=True)
        t1.start()
        t2.start()

        while st.session_state.get("running", False):
            ret, frame = cap.read()
            if not ret:
                break

            if enable_sign:
                frame = pipeline.predict_sign(frame)

            state = pipeline.get_state()
            frame = render_overlay(frame, state)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_column_width=True)

            text_box.markdown(f"**Speech:** {state['text'] or '—'}")
            sign_box.markdown(f"**Sign:** {state['sign'] or '—'}")
            audio_box.markdown(f"**Audio:** {state['audio_event'] or '—'}")

            time.sleep(0.03)

        cap.release()

    if stop:
        st.session_state["running"] = False
        st.success("Stopped")

if __name__ == "__main__":
    main()