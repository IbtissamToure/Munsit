# Munsit

An AI-powered system that translates Arabic speech and sign language in real time to support deaf and hard-of-hearing students in Arab universities.

## The Problem
Deaf students in Arab universities attend lectures and pass exams, but remain excluded from discussions, events, and social interactions due to a single barrier the language of the moment.

## The Solution
A smart system combining three AI modules:
- **Speech to Text** — converts spoken Arabic to text using Whisper
- **Sign Language Detection** — recognizes Arabic sign language using MediaPipe + LSTM
- **Environmental Audio** — classifies surrounding sounds (applause, laughter, music) and converts them to visual signals using CNN

## Tech Stack
| Tool | Purpose |
|------|---------|
| OpenAI Whisper | Arabic speech recognition |
| MediaPipe | Hand landmark detection |
| PyTorch | LSTM + CNN model training |
| OpenCV | Real-time video processing |
| Streamlit | Web interface |

## Project Status
🚧 Under active development — Day 2 of 14
