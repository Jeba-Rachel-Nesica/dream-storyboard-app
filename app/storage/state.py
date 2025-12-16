import streamlit as st
import json
from app.pipelines.utils import save_json
from app.storage import paths
from datetime import datetime

# Session state helpers

def init_session():
    if 'identity' not in st.session_state:
        st.session_state['identity'] = None
    if 'script' not in st.session_state:
        st.session_state['script'] = None
    if 'beats' not in st.session_state:
        st.session_state['beats'] = None
    if 'keyframes' not in st.session_state:
        st.session_state['keyframes'] = None
    if 'styled_frames' not in st.session_state:
        st.session_state['styled_frames'] = None

def save_identity(info):
    st.session_state['identity'] = info

def get_identity():
    return st.session_state.get('identity')

def save_script(text):
    st.session_state['script'] = text

def get_script():
    return st.session_state.get('script')

def save_beats(beats):
    st.session_state['beats'] = beats
    save_json(beats, paths.get_beats_path())

def get_beats():
    return st.session_state.get('beats')

def save_keyframes(frames):
    st.session_state['keyframes'] = frames
    # Save images and metadata
    meta = {
        'keyframes': [],
        'timestamp': datetime.now().isoformat(),
    }
    for i, kf in enumerate(frames):
        path = paths.get_keyframe_path(i)
        kf['img'].save(path)
        meta['keyframes'].append({
            'path': path,
            'seed': kf['seed'],
            'prompt': kf['prompt'],
            'negative_prompt': kf['negative_prompt'],
        })
    save_json(meta, paths.get_metadata_path())

def get_keyframes():
    return st.session_state.get('keyframes')

def save_styled_frames(frames):
    st.session_state['styled_frames'] = frames
    for i, img in enumerate(frames):
        img.save(paths.get_styled_path(i))

def get_styled_frames():
    return st.session_state.get('styled_frames')

def delete_outputs():
    paths.delete_outputs()
    for k in ['keyframes', 'styled_frames']:
        st.session_state[k] = None
