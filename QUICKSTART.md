# Quick Start Guide

## Installation (5 minutes)

### Windows
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test imports
python test_imports.py
```

### Linux/Mac
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test imports
python test_imports.py
```

## Running the App

### Option 1: Using Run Script (Recommended)
```bash
python run_streamlit.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run app/main.py
```

### Option 3: Python Module
```bash
python -m streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## First Time Usage

### 1. Upload Photos (Step 1)
- Upload 3-10 clear photos of a person
- Photos should be well-lit, front-facing
- Only one face per photo
- Check consent checkbox

**Expected**: Face detection runs, shows best reference face

### 2. Upload Script (Step 2)
- Upload a .txt file with rewritten dream script
- Script should be at least 40 characters
- Use calm, comforting language

**Expected**: Script preview appears

### 3. Edit Beats (Step 3)
- Click "Regenerate beats" to extract scenes
- Edit prompts, emotions, camera hints
- Ensure all content is safe

**Expected**: 4-8 beats extracted, editable

### 4. Generate Frames (Step 4)
⚠️ **This step is SLOW on CPU (5-10 min per beat)**
- 3 candidates generated per beat
- Ranked by face similarity
- Click "Select" to choose best

**Expected**: Images generated with face consistency

### 5. Export Video (Step 5)
- Enter style preset (e.g., "abstract dreamy watercolor")
- Stylization applies to each frame
- Click "Compose & Preview Video"
- Download MP4 when ready

**Expected**: Video with crossfades and captions

## Troubleshooting

### ❌ "Import cv2 could not be resolved"
```bash
pip install opencv-python
```

### ❌ "CUDA out of memory"
- CPU fallback is automatic
- Close other applications
- Reduce candidates (edit frame_generator.py: n=3 -> n=2)

### ❌ "InsightFace model not found"
- Models download on first run (~1GB)
- Check internet connection
- Wait 5-10 minutes for download

### ❌ "FFmpeg not found"
**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract to C:\ffmpeg
3. Add C:\ffmpeg\bin to PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### ❌ App is very slow
- **GPU**: Install CUDA toolkit for faster generation
- **CPU**: Reduce candidates, expect 5-10 min per beat
- **RAM**: Close other applications

## Features

✅ **Local-first**: All processing on your machine, no cloud uploads
✅ **Safety filters**: Blocks unsafe/inappropriate content  
✅ **Face consistency**: Uses InsightFace for identity preservation
✅ **Customizable**: Edit prompts, styles, emotions
✅ **Export**: Download MP4 with captions and effects

## System Requirements

### Minimum
- Python 3.11+
- 8GB RAM
- 10GB disk space
- CPU only (slow)

### Recommended
- Python 3.11+
- 16GB RAM
- 20GB disk space
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8+

## Need Help?

1. Check `FIXES_SUMMARY.md` for common errors
2. Check `STREAMLIT_README.md` for architecture
3. Check `DEPLOYMENT_CHECKLIST.md` for cloud deployment
4. Review code comments in `app/` directory

## Example Workflow

```
1. Prepare 5 photos of yourself (clear, well-lit)
2. Write a calm dream script (save as dream.txt)
3. Run: python run_streamlit.py
4. Upload photos → Upload script → Generate beats
5. Generate 3 candidates per beat (15-30 min on CPU)
6. Select best candidates → Stylize → Export video
7. Download rehearsal_video.mp4
```

## What to Expect

- **First run**: Model downloads (~5GB for Stable Diffusion)
- **Photo upload**: Instant face detection
- **Beat extraction**: ~1 second
- **Frame generation**: 
  - GPU: 10-30 seconds per candidate
  - CPU: 5-10 minutes per candidate
- **Video composition**: 30-60 seconds

## Tips

💡 Use well-lit, front-facing photos for best results  
💡 Keep scripts calm and positive (safety filters active)  
💡 Edit prompts to match your vision  
💡 CPU mode works but is much slower  
💡 Close other apps to free up RAM/GPU memory  

---

**Ready to start?** Run `python run_streamlit.py`
