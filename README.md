# Speak Portrait - ML Pipeline

This repository contains the core machine learning pipeline models for the SpeakPortrait SaaS application. It was extracted from the original backend repository to maintain separation of concerns between business logic and ML experimentation.

## Contents
- `Age1.ipynb`: Model experimentation and inference script for the StyleGAN-based age transformation pipeline.
- `Carve.ipynb`: Model experimentation script for background carving and face landmark extraction.
- `Zonos_1.ipynb`: Zonos TTS integration and audio processing experimentation.

## Notes
These models are currently structured as Jupyter Notebooks. Future steps may include wrapping them into a dedicated inference API (such as FastAPI) and generating a standardized `requirements.txt`.
