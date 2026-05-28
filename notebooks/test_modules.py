import whisper
import mediapipe
import torch
import librosa
import cv2
import os

print("Whisper:", "OK")
print("MediaPipe:", "OK")
print("PyTorch:", torch.__version__)
print("Sign model:", "OK" if os.path.exists("models/sign_classifier.pth") else "MISSING")
print("Audio model:", "OK" if os.path.exists("models/audio_cnn.pth") else "MISSING")