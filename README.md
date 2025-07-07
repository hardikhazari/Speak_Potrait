# Speak Portrait - ML Pipeline

Welcome to the **Speak Portrait ML Pipeline** repository. This is an extracted module of the larger **Speak Portrait SaaS**, designed to handle the core artificial intelligence operations: AI talking-head video generation, background carving, age transformation, and Text-to-Speech (TTS).

---

## 🌐 Full Project Context

**Speak Portrait** is a comprehensive multimedia SaaS platform that allows users to generate and manage dynamic "talking portrait" videos. It features secure user authentication, robust cloud storage, and real-time processing pipelines. 

To view the full stack, please refer to the main application repositories:
*   🔗 **[Backend API Repository (Node.js/Express)](https://github.com/yasharyasaxena/speak_portrait_backend.git)**: Manages PostgreSQL via Prisma, AWS S3 buckets for media storage, and Firebase authentication.
*   🔗 **Frontend Application (Next.js/React)**: The main user interface, featuring Tailwind styling, Firebase client auth, and real-time WebSocket integrations for tracking generation status.

---

## 🧠 The ML Architecture

To maintain a clean separation of concerns and avoid entangling heavy deep learning dependencies with standard Express HTTP servers, the ML pipeline has been isolated into this repository.

This repository focuses exclusively on inference and data processing using PyTorch. 

### Core Modules:
1.  **`age_transformation.py`**: Utilizes a StyleGAN-based architecture (via pSp and SAM) paired with `dlib` 68-point landmarks to dynamically alter the perceived age of a subject's face while retaining their identity.
2.  **`background_carving.py`**: Integrates `carvekit` (using the Tracer-B7 network and FBA matting) to perform highly accurate background removal and replacement, handling difficult edge cases like hair seamlessly.
3.  **`zonos_tts.py`**: Leverages the **Zonos transformer** for zero-shot text-to-speech generation. It extracts speaker embeddings from reference audio to clone voices dynamically.

---

## 🛠️ Setup & Installation

To run these scripts locally or package them into a dedicated inference server (like FastAPI), follow these steps:

### 1. Prerequisites
You must have Python 3.10+ installed and a CUDA-capable GPU is highly recommended for optimal processing speed. 

### 2. Install Dependencies
Install the required packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

*Note on external dependencies:*
*   **SAM Repository**: For the age transformation model to work, you must clone the SAM repository:
    ```bash
    git clone https://github.com/yuval-alaluf/SAM.git
    ```
*   **Zonos**: For TTS, install via `uv` or clone the repository directly:
    ```bash
    git clone https://github.com/Zyphra/Zonos.git
    cd Zonos && uv sync
    ```

### 3. Model Weights
You will need to download the appropriate pre-trained `.pt` / `.dat` weight files and place them in your working directory:
- `sam_ffhq_aging.pt` (SAM Age Transformer)
- `shape_predictor_68_face_landmarks.dat` (Dlib)
- CarveKit weights will download automatically on first run.

---

## 🚀 Usage

Each module is designed as an importable class, but can also be tested via the command line.

**Example: Background Replacement**
```python
from background_carving import BackgroundReplacementModel

carver = BackgroundReplacementModel()
result = carver.replace_background(
    foreground_url="http://example.com/user.jpg",
    background_url="http://example.com/bg.jpg",
    output_path="final_composite.png"
)
```

**Example: Text-to-Speech**
```python
from zonos_tts import ZonosTTSModel

tts = ZonosTTSModel(reference_audio_path="user_voice_sample.mp3")
tts.generate_speech(text="Hello world!", output_path="output.wav")
```
