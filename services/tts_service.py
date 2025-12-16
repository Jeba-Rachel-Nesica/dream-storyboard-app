import os, uuid, pyttsx3
from config import Config

_engine = None
def _engine_init():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 165)
    return _engine

def synth_to_file(text: str) -> str:
    os.makedirs(Config.MEDIA_DIR, exist_ok=True)
    fname = f"tts_{uuid.uuid4().hex}.wav"
    fpath = os.path.join(Config.MEDIA_DIR, fname)
    eng = _engine_init()
    eng.save_to_file(text, fpath)
    eng.runAndWait()
    return fpath
