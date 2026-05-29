# Munsit

An AI-powered system that translates Arabic speech and sign language in real time to support deaf and hard-of-hearing students in Arab universities.

## Motivation
Deaf and hard-of-hearing students in Arab universities face a persistent communication barrier — not in academics, but in everything around it. Lectures are accessible, but real-time conversations, group discussions, and social events are not. Existing solutions address translation in isolation; Munsit integrates speech recognition, sign language detection, and environmental audio analysis into a single real-time system designed for Arabic-speaking contexts.

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
🚧 Under active development — Day 6 of 14
