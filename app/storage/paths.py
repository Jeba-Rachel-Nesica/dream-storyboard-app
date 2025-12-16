import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(BASE_DIR, 'outputs')
KEYFRAME_DIR = os.path.join(DATA_DIR, 'keyframes')
STYLED_DIR = os.path.join(DATA_DIR, 'styled_frames')
META_PATH = os.path.join(DATA_DIR, 'metadata.json')
BEATS_PATH = os.path.join(DATA_DIR, 'beats.json')
VIDEO_PATH = os.path.join(DATA_DIR, 'rehearsal_video.mp4')

os.makedirs(KEYFRAME_DIR, exist_ok=True)
os.makedirs(STYLED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_keyframe_path(idx):
    return os.path.join(KEYFRAME_DIR, f'beat_{idx+1:02d}.png')

def get_styled_path(idx):
    return os.path.join(STYLED_DIR, f'beat_{idx+1:02d}.png')

def get_metadata_path():
    return META_PATH

def get_beats_path():
    return BEATS_PATH

def get_video_path():
    return VIDEO_PATH

def delete_outputs():
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(KEYFRAME_DIR, exist_ok=True)
    os.makedirs(STYLED_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
