"""
Test script to verify all imports work correctly
Run this before deploying to catch any import errors
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Testing imports...")
print("-" * 50)

try:
    print("✓ Testing core libraries...")
    import numpy as np
    import torch
    from PIL import Image
    print("  ✓ numpy, torch, PIL")
    
    print("\n✓ Testing computer vision...")
    import cv2
    print(f"  ✓ opencv-python (cv2) version: {cv2.__version__}")
    
    print("\n✓ Testing web frameworks...")
    import streamlit as st
    print(f"  ✓ streamlit")
    
    print("\n✓ Testing ML frameworks...")
    from diffusers import StableDiffusionPipeline
    from transformers import GPT2Tokenizer
    print("  ✓ diffusers, transformers")
    
    print("\n✓ Testing face recognition...")
    from insightface.app import FaceAnalysis
    print("  ✓ insightface")
    
    print("\n✓ Testing video processing...")
    import moviepy.editor as mpy
    print("  ✓ moviepy")
    
    print("\n✓ Testing app modules...")
    from app.pipelines import beats, safety, identity, generation, stylize, compose_video, utils
    print("  ✓ app.pipelines.*")
    
    from app.storage import state, paths
    print("  ✓ app.storage.*")
    
    from app.ui import upload, script_input, beats_editor, frame_generator, video_export
    print("  ✓ app.ui.*")
    
    print("\n" + "=" * 50)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("=" * 50)
    print("\nYou can now run the app with:")
    print("  python run_streamlit.py")
    print("  OR")
    print("  streamlit run app/main.py")
    
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("\nPlease install missing dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    sys.exit(1)
