import threading
import numpy as np
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import whisper
import sounddevice as sd
import soundfile as sf
import librosa
import tempfile
import os
import pyttsx3
import time
from dotenv import load_dotenv

load_dotenv()

def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    t = threading.Thread(target=run, daemon=True)
    t.start()

CLASSES = ['ALIF', 'BAA', 'TA', 'THA', 'JEEM', 'HAA', 'KHAA',
           'DELL', 'DHELL', 'RAA', 'ZAY', 'SEEN', 'SHEEN', 'SAD',
           'DAD', 'TAA', 'DHAA', 'AYN', 'GHAYN', 'FAA', 'QAAF',
           'KAAF', 'LAAM', 'MEEM', 'NOON', 'HA', 'WAW', 'YA']

AUDIO_CLASSES = ['street_music', 'children_playing', 'siren', 'car_horn', 'dog_bark']

class SignClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 28)
        )
    def forward(self, x):
        return self.network(x)

class AudioCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.conv_layers(x))

class MunsitPipeline:
    def __init__(self):
        self.device = torch.device("cpu")

        self.whisper_model = whisper.load_model(
            os.getenv("WHISPER_MODEL", "base")
        )

        self.sign_model = SignClassifier().to(self.device)
        self.sign_model.load_state_dict(
            torch.load("models/sign_classifier.pth", map_location=self.device)
        )
        self.sign_model.eval()

        self.audio_model = AudioCNN().to(self.device)
        self.audio_model.load_state_dict(
            torch.load("models/audio_cnn.pth", map_location=self.device)
        )
        self.audio_model.eval()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.current_text = ""
        self.current_sign = ""
        self.current_audio_event = ""
        self.lock = threading.Lock()
        self.last_sign = ""
        self.last_sign_time = 0

    def transcribe_audio(self, duration=5):
        sample_rate = 16000
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            result = self.whisper_model.transcribe(f.name, language="ar")
        with self.lock:
            self.current_text = result["text"]

    def predict_sign(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            x = torch.FloatTensor(landmarks).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.sign_model(x).argmax(dim=1).item()
            current_time = time.time()
            if CLASSES[pred] != self.last_sign or current_time - self.last_sign_time > 1.5:
                with self.lock:
                    self.current_sign = CLASSES[pred]
                speak(CLASSES[pred])
                self.last_sign = CLASSES[pred]
                self.last_sign_time = current_time
            self.mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )
        return frame

    def classify_audio_event(self, duration=4):
        sample_rate = 22050
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        x = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.audio_model(x).argmax(dim=1).item()
        with self.lock:
            self.current_audio_event = AUDIO_CLASSES[pred]

    def get_state(self):
        with self.lock:
            return {
                "text": self.current_text,
                "sign": self.current_sign,
                "audio_event": self.current_audio_event
            }

if __name__ == "__main__":
    print("Loading pipeline...")
    pipeline = MunsitPipeline()
    print("Pipeline loaded successfully")
    state = pipeline.get_state()
    print("State:", state)