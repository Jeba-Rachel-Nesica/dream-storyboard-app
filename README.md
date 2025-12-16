# Dream Storyboard Video Generator (Phase-2)

**Demo/Research Prototype – Not Medical Advice**

## Overview
This app generates a personalized, abstract video storyboard from a set of person photos and a rewritten dream script. It is designed for research and demonstration only. No therapy modules are included.

**Pipeline:**
1. Upload 3–10 photos (identity reference)
2. Upload a rewritten dream script (text)
3. Extract 4–8 beats (scenes) from the script
4. Generate identity-consistent images for each beat
5. Stylize images into abstract/dreamy frames
6. Stitch frames into an MP4 video with crossfades and captions

**Safety/Ethics:**
- Consent checkbox: “I have permission to use these photos.”
- Safety filter blocks unsafe/graphic/illegal content
- All files stored locally only
- “Delete project outputs” button
- Disclaimer: demo/research prototype; not medical advice

## Quickstart (Windows)

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Requirements
- Python 3.11+
- Windows 10/11 (tested)
- NVIDIA GPU recommended for image generation (CPU fallback available, slower)

## Features
- Local-first, no cloud upload
- Streamlit UI: upload, edit, preview, download
- Pluggable diffusion backend (local SD via diffusers, or Ollama/ComfyUI endpoint)
- Face embedding/identity conditioning (InsightFace or fallback)
- Abstract stylization (img2img)
- Video composition (FFmpeg or moviepy)
- Full reproducibility: metadata.json

## Folder Structure
```
/app
  main.py
  ui/
    upload.py
    script_input.py
    beats_editor.py
    frame_generator.py
    video_export.py
  pipelines/
    beats.py
    safety.py
    identity.py
    generation.py
    stylize.py
    compose_video.py
    utils.py
  storage/
    paths.py
    state.py
/tests
  test_beats.py
  test_safety.py
  test_compose_video.py
README.md
requirements.txt
.env.example
```

## Notes
- All outputs are local. No data leaves your machine.
- If identity adapters are unavailable, fallback to img2img with high face-preservation.
- GPU recommended for reasonable speed.
- For research/demo only. Not for clinical or diagnostic use.

## Disclaimer
This is a research prototype. It is not intended for medical use or advice. All content is generated for demonstration purposes only. Use responsibly and ethically.
