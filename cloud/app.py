import streamlit as st

st.set_page_config(
    page_title="Munsit",
    page_icon="👁",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0a0f1a; }
    h1, h2 { color: #00c9ff; }
    .metric-card {
        background: #111d2c;
        border: 1px solid #1a2d42;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("Munsit 👁")
st.caption("Real-Time Arabic Accessibility System")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2>🎙️</h2>
        <h3 style="color:#00c9ff">Speech Recognition</h3>
        <p style="color:#5a7a9a">OpenAI Whisper<br>Arabic language support<br>Real-time transcription</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2>✋</h2>
        <h3 style="color:#7b5ea7">Sign Detection</h3>
        <p style="color:#5a7a9a">MediaPipe + Neural Network<br>28 Arabic letters<br>84.2% accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h2>🔊</h2>
        <h3 style="color:#00e5a0">Audio Classification</h3>
        <p style="color:#5a7a9a">CNN model<br>5 sound categories<br>88.5% accuracy</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("How It Works")
    st.markdown("""
    1. **Camera** captures hand movements → MediaPipe extracts 21 landmarks
    2. **Neural network** classifies the sign → letter spoken aloud via TTS
    3. **Microphone** captures speech → Whisper transcribes to Arabic text
    4. **Audio CNN** classifies environmental sounds → visual alert displayed
    """)

with col2:
    st.subheader("Model Performance")
    st.metric("Sign Language Accuracy", "84.2%", "28 Arabic letters")
    st.metric("Audio Classification", "88.5%", "5 sound categories")
    st.metric("Training Samples", "4,507", "sign language dataset")

st.divider()
st.subheader("Run Locally")
st.code("""
git clone https://github.com/IbtissamToure/Munsit.git
cd Munsit
python -m venv munsit_env
source munsit_env/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
""", language="bash")

st.divider()
st.caption("Built with Python · Whisper · MediaPipe · PyTorch · Streamlit")