# services/tts_utils.py
import base64
import tempfile
import os

def _b64_of(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def tts_base64(text: str) -> dict:
    """
    Return {"b64": "...", "mime": "audio/mpeg|audio/wav"}.
    Tries gTTS (mp3, online). Falls back to pyttsx3 (wav, offline).
    If both fail, returns {"b64": "", "mime": ""}.
    """
    text = (text or "").strip()
    if not text:
        return {"b64": "", "mime": ""}

    # 1) Try gTTS (mp3)
    try:
        from gtts import gTTS  # may raise ImportError
        mp3_path = tempfile.mkstemp(suffix=".mp3")[1]
        gTTS(text).save(mp3_path)
        return {"b64": _b64_of(mp3_path), "mime": "audio/mpeg"}
    except Exception:
        pass

    # 2) Try offline pyttsx3 (wav)
    try:
        import pyttsx3
        wav_path = tempfile.mkstemp(suffix=".wav")[1]
        engine = pyttsx3.init()  # SAPI5 (Windows) / NSSpeechSynthesizer (macOS) / eSpeak (Linux)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        # Some engines are async; ensure file exists and has content
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            return {"b64": _b64_of(wav_path), "mime": "audio/wav"}
    except Exception:
        pass

    # 3) Give up; caller/UI can fall back to browser TTS
    return {"b64": "", "mime": ""}
