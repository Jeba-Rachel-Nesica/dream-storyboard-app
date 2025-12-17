import os
import numpy as np
try:
    import moviepy.editor as mpy
    from moviepy.video.fx import all as vfx
except ImportError:
    import moviepy.editor as mpy
from app.storage import paths

def compose_video(frames, beats, out_path=None, duration=2.0, crossfade=0.4):
    from PIL import Image
    clips = []
    for i, img in enumerate(frames):
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        clip = mpy.ImageClip(img_array).set_duration(duration)
        
        # Ken Burns effect: slow zoom
        try:
            clip = clip.fx(mpy.vfx.resize, lambda t: 1 + 0.05 * (t / duration))
        except Exception:
            pass  # Skip if zoom fails
        
        # Optional: add caption
        txt = beats[i]['scene_text'] if i < len(beats) else ''
        if txt:
            try:
                txt_clip = mpy.TextClip(txt, fontsize=32, color='white', bg_color='rgba(0,0,0,0.4)', method='caption', size=(clip.w, None)).set_duration(duration)
                txt_clip = txt_clip.set_position(('center', 'bottom'))
                clip = mpy.CompositeVideoClip([clip, txt_clip])
            except Exception:
                pass  # Skip captions if TextClip fails
        clips.append(clip)
    
    video = mpy.concatenate_videoclips(clips, method="compose", padding=-crossfade if crossfade > 0 else 0)
    out_path = out_path or paths.get_video_path()
    video.write_videofile(out_path, fps=24, codec='libx264', audio=False)
    return out_path
