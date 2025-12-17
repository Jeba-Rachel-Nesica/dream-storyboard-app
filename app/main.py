import streamlit as st
from ui import upload, script_input, beats_editor, frame_generator, video_export
from storage import state
from pipelines.safety import show_disclaimer

st.set_page_config(page_title="Dream Storyboard Video Generator", layout="wide")

show_disclaimer()

# Session state
state.init_session()

st.title("Dream Storyboard Video Generator")

# 1. Upload photos + consent
upload.upload_photos()

# 2. Upload rewritten script
script_input.script_input()

# 3. Extract/edit beats
beats_editor.beats_editor()

# 4. Generate/select keyframes
frame_generator.frame_generator()

# 5. Stylize + compose video
video_export.video_export()
