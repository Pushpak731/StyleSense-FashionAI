# StyleSense — FashionAI 👗

**StyleSense** is an AI-powered personalized fashion recommendation system. Upload your photos and it analyzes your **skin tone, eye color, hair color, and body shape**, then recommends outfits, color palettes, fabrics, and styling advice — backed by an LLM-powered stylist chatbot.

## How It Works

The project has two components:

1. **`code2_modified.py` — PersonalAnalyzer** (computer vision + ML)
   - Uses **MediaPipe FaceMesh** (with refined iris landmarks) to detect facial regions
   - Extracts dominant skin tone, under-tone, eye color, and hair color from images via clustering
   - Classifies body shape / proportions
   - An **XGBoost** multi-output classifier maps extracted features to personalized recommendations (clothing colors, fits, materials, patterns, jewelry metals, shoes) using the curated knowledge table in `_recommendations_with_hex.csv`

2. **`chatbot_ui.py` — Stylist Chatbot** (Streamlit + Groq LLM)
   - Polished Streamlit UI for uploading photos and viewing your personalized palette
   - An LLM chatbot (Groq API) that answers styling questions informed by your analysis results

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your Groq API key

In `chatbot_ui.py`, replace the placeholder with your key (get one at [console.groq.com](https://console.groq.com)):

```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

> ⚠️ Tip: for anything public, load the key from an environment variable instead of hardcoding it.

### 3. Run the app

```bash
streamlit run chatbot_ui.py
```

Upload a front-facing photo, let the analyzer extract your features, and start chatting with your AI stylist.

## What You Get

- Personalized **clothing color palette** (with hex codes) and colors to avoid
- Recommended **fits, fabrics, and patterns** for your body shape
- Jewelry metal and shoe style suggestions
- A conversational **stylist chatbot** for follow-up fashion advice

## Requirements

Python 3.9–3.11. Key dependencies: `mediapipe`, `opencv-python`, `scikit-learn`, `xgboost`, `streamlit`, `groq`, `Pillow` — see `requirements.txt` for the full pinned list.
