import os
import moviepy.editor as mpy
from app.storage import paths

def compose_video(frames, beats, out_path=None, duration=2.0, crossfade=0.4):
    clips = []
    for i, img in enumerate(frames):
        clip = mpy.ImageClip(img).set_duration(duration)
        # Ken Burns effect: slow zoom
        clip = clip.fx(mpy.vfx.zoom_in, 1.05)
        # Optional: add caption
        txt = beats[i]['scene_text'] if i < len(beats) else ''
        if txt:
            txt_clip = mpy.TextClip(txt, fontsize=32, color='white', bg_color='rgba(0,0,0,0.4)').set_duration(duration)
            txt_clip = txt_clip.set_position(('center', 'bottom')).margin(bottom=30, opacity=0)
            clip = mpy.CompositeVideoClip([clip, txt_clip])
        clips.append(clip)
    video = mpy.concatenate_videoclips(clips, method="compose", padding=-crossfade)
    out_path = out_path or paths.get_video_path()
    video.write_videofile(out_path, fps=24, codec='libx264', audio=False)
    return out_path
