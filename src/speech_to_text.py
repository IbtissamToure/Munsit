import whisper
import sounddevice as sd
import numpy as np
import tempfile 
import soundfile as sf
from dotenv import load_dotenv
import os
load_dotenv()
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
model = whisper.load_model(MODEL_SIZE)
def record_audio (duration=6 , sample_rate = 16000):
    print(f"Recording for{duration} seconds..")
    audio = sd.rec(
        int(duration*sample_rate),
        samplerate = sample_rate,
        channels = 1,
        dtype = 'float32'
    )
    sd.wait()
    print("Recording Done")
    return audio.flatten()
def transcribe(audio_array , sample_rate=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_array, sample_rate)
        result = model.transcribe(f.name, language ="ar")
    return result["text"]
def listen_and_transcribe(duration =5):
    audio = record_audio(duration)
    text = transcribe(audio)
    return text

if __name__ == "__main__" :
    print("MUNSIT Speech to text")
    print("="*40)
    text = listen_and_transcribe(duration = 5)
    print(f"\ntext: {text}")



