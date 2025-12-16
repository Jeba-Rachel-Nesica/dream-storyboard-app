# services/asr_utils.py
import numpy as np
import torch
from transformers import pipeline

_ASR = None
MODEL_NAME = "openai/whisper-small.en"  # or "openai/whisper-base.en" for faster CPU

def get_asr():
    """Lazy-load Whisper (English-only)."""
    global _ASR
    if _ASR is not None:
        return _ASR

    device = 0 if torch.cuda.is_available() else -1

    # Keep generation args minimal and widely supported.
    gen_kwargs = {
        "temperature": 0.0,  # deterministic decoding
    }
    # If your Transformers version supports beams here, you can enable:
    # gen_kwargs["num_beams"] = 5

    _ASR = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        generate_kwargs=gen_kwargs,
    )
    return _ASR

def preload_asr():
    """Warm model at boot to avoid first-call lag."""
    asr = get_asr()
    dummy = {"array": np.zeros(8000, dtype=np.float32), "sampling_rate": 16000}
    try:
        asr(dummy, chunk_length_s=5)
    except Exception:
        # Any warmup hiccup shouldn't block app startup
        pass

def run_asr_from_array(audio_array: np.ndarray, sampling_rate: int = 16000, max_seconds: int = 20) -> str:
    """
    Transcribe a numpy audio array (mono) with a hard duration cap.
    Returns plain text (no timestamps).
    """
    asr = get_asr()

    if not isinstance(audio_array, np.ndarray):
        audio_array = np.array(audio_array, dtype=np.float32)
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Cap duration to avoid long stalls
    max_len = int(max_seconds * sampling_rate)
    if audio_array.shape[0] > max_len:
        audio_array = audio_array[:max_len]

    payload = {"array": audio_array, "sampling_rate": sampling_rate}

    out = asr(
        payload,
        chunk_length_s=12,            # shorter chunks improve stability on short clips
        stride_length_s=(4, 4),       # small overlap for smoother boundaries
        return_timestamps=False,
    )
    return (out.get("text") or "").strip()
