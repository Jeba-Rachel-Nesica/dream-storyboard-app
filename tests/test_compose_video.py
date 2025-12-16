from app.pipelines.compose_video import compose_video
from PIL import Image
import numpy as np
import os

def test_compose_video_tmp(tmp_path):
    # Create dummy images
    imgs = [Image.fromarray(np.uint8(np.random.rand(256,256,3)*255)) for _ in range(4)]
    beats = [{'scene_text': f'Beat {i+1}'} for i in range(4)]
    out_path = tmp_path / "test_video.mp4"
    path = compose_video(imgs, beats, out_path=str(out_path))
    assert os.path.exists(path)
    assert path.endswith(".mp4")
