# Running the Dream Storyboard Video Generator (Streamlit App)

## Local Development

### Prerequisites
- Python 3.11+
- NVIDIA GPU recommended (CPU fallback available)

### Setup

1. **Create and activate virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
# Option 1: Using the run script
python run_streamlit.py

# Option 2: Direct streamlit command
streamlit run app/main.py

# Option 3: Using Python module
python -m streamlit run app/main.py
```

## Deployment to Streamlit Cloud

### Required Files
- `requirements.txt` - Python dependencies
- `packages.txt` - System-level dependencies (for Linux)
- `.streamlit/config.toml` - Streamlit configuration

### Steps

1. **Push to GitHub repository**
2. **Connect to Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Connect your GitHub repository
   - Set main file path: `app/main.py`
   - Deploy

3. **Environment Variables (if needed):**
   - Add any API keys or secrets in Streamlit Cloud dashboard
   - Go to "Manage app" > "Settings" > "Secrets"

## Troubleshooting

### ModuleNotFoundError: No module named 'cv2'
- Ensure `opencv-python` is in requirements.txt
- For cloud deployment, `opencv-python-headless` is recommended
- System dependencies are in `packages.txt`

### ImportError: Cannot import name 'X'
- Verify all `__init__.py` files exist in app directories
- Check Python path includes project root
- Use absolute imports: `from app.module import ...`

### InsightFace model download issues
- First run downloads models (~1GB)
- Ensure internet connection and sufficient disk space
- Models are cached in `~/.insightface/`

### MoviePy / FFmpeg errors
- Ensure FFmpeg is installed (system-level)
- Windows: Download from https://ffmpeg.org/
- Linux: `sudo apt-get install ffmpeg`
- Already included in `packages.txt` for cloud deployment

### Stable Diffusion OOM (Out of Memory)
- Reduce batch size in generation
- Use CPU if GPU memory is insufficient
- Consider using smaller models

## Architecture

```
app/
├── main.py              # Entry point
├── ui/                  # UI modules
│   ├── upload.py        # Photo upload
│   ├── script_input.py  # Script input
│   ├── beats_editor.py  # Beat editing
│   ├── frame_generator.py  # Frame generation
│   └── video_export.py  # Video composition
├── pipelines/           # Processing pipelines
│   ├── beats.py         # Beat extraction
│   ├── safety.py        # Content safety
│   ├── identity.py      # Face detection
│   ├── generation.py    # Image generation
│   ├── stylize.py       # Image stylization
│   └── compose_video.py # Video composition
└── storage/             # State management
    ├── state.py         # Session state
    └── paths.py         # File paths
```

## Dependencies

Key packages:
- `streamlit` - Web UI framework
- `opencv-python` - Image processing
- `insightface` - Face detection/recognition
- `diffusers` - Stable Diffusion models
- `torch` - Deep learning framework
- `moviepy` - Video composition
- `profanity-check` - Content safety

See `requirements.txt` for full list.
