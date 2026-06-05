# Munsit — Real-Time Arabic Accessibility System
An AI-powered system that bridges communication gaps for deaf and hard-of-hearing individuals in Arabic-speaking environments. Combines three independent AI modules into a unified real-time pipeline.

## Live Demo
 [munsit-lzmhfu6lugfpfbwctqcf2n.streamlit.app](https://munsit-lzmhfu6lugfpfbwctqcf2n.streamlit.app)

> Full real-time system (camera + microphone + TTS) runs locally only. The link above shows project overview and model metrics.

## System Architecture
Input               Processing                Output
─────               ──────────                ──────
Microphone    →     Whisper (ASR)         →   Arabic text overlay
Camera        →     MediaPipe + FC Net    →   Letter spoken aloud
Environment   →     Audio CNN             →   Sound event alert

## Modules

### 1. Speech Recognition
- Model: OpenAI Whisper (base)
- Language: Arabic
- Pipeline: sounddevice → Whisper → arabic-reshaper → overlay

### 2. Sign Language Detection
- Architecture: Fully connected network (63 → 128 → 64 → 28)
- Input: 21 hand landmarks × XYZ = 63 features via MediaPipe
- Dataset: Arabic Sign Language — 4,507 samples, 28 classes
- Validation Accuracy: **84.2%**
- Output: Predicted letter displayed + spoken via TTS

### 3. Environmental Audio Classification
- Architecture: CNN (3 conv layers + AdaptiveAvgPool)
- Input: Mel Spectrogram (64 bands, 4s window) via librosa
- Dataset: UrbanSound8K — 5 classes
- Validation Accuracy: **88.5%**
- Classes: street_music, children_playing, siren, car_horn, dog_bark

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hands |
| Speech Recognition | OpenAI Whisper |
| Model Training | PyTorch 2.2.1 |
| Audio Features | Librosa |
| Text to Speech | pyttsx3 |
| Interface | Streamlit |
| Language | Python 3.11 |

## Run Locally
```bash
git clone https://github.com/IbtissamToure/Munsit.git
cd Munsit
python -m venv munsit_env
source munsit_env/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

## Project Structure
munsit/
├── src/
│   ├── app.py               # Streamlit interface
│   ├── pipeline.py          # Unified pipeline class
│   ├── speech_to_text.py    # Whisper STT module
│   ├── sign_detection.py    # MediaPipe hand tracking
│   ├── train_model.py       # Sign classifier training
│   ├── audio_classifier.py  # Audio CNN training
│   ├── prepare_dataset.py   # Dataset preprocessing
│   └── visual_overlay.py    # Frame rendering
├── models/
│   ├── sign_classifier.pth  # Trained sign model
│   └── audio_cnn.pth        # Trained audio model
├── cloud/
│   └── app.py               # Cloud deployment version
└── requirements.txt

## Roadmap
- [x] Environment setup
- [x] Arabic speech recognition (Whisper)
- [x] Hand landmark detection (MediaPipe)
- [x] Sign language classifier — 84.2%
- [x] Audio event classifier — 88.5%
- [x] Unified real-time pipeline
- [x] Streamlit interface + TTS
- [x] Cloud deployment
- [ ] Word-level sign language recognition
- [ ] Meta Quest 3 integration

done by: Ibtissam Toure 