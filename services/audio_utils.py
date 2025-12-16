import base64, subprocess, tempfile, numpy as np
from scipy.io import wavfile
import imageio_ffmpeg

def _trim_silence(arr: np.ndarray, sr: int, frame_ms: int = 30, thresh_db: float = -35.0):
    """
    Simple leading/trailing silence trim by frame energy.
    Keeps mid content intact (no aggressive VAD).
    """
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    frame = int(sr * frame_ms / 1000)
    frame = max(frame, 1)
    # pad to full frames
    n = int(np.ceil(len(arr) / frame)) * frame
    pad = n - len(arr)
    if pad:
        arr = np.pad(arr, (0, pad))
    arr2 = arr.reshape(-1, frame)
    # RMS in dBFS
    rms = np.sqrt((arr2 ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # find first/last above threshold
    nz = np.where(db > thresh_db)[0]
    if nz.size == 0:
        return arr[:0]  # all silence
    i0, i1 = nz[0] * frame, (nz[-1] + 1) * frame
    return arr[i0:i1]

def ffmpeg_to_wav_array_from_b64(b64_audio: str, mime_hint: str = "audio/webm"):
    """
    Base64 -> ffmpeg -> 16k mono WAV -> float32 [-1,1], with gentle leading/trailing silence trim.
    No EQ/denoise that can mangle consonants.
    """
    ext = ".webm" if "webm" in (mime_hint or "").lower() else ".ogg"
    raw_path = tempfile.mkstemp(suffix=ext)[1]
    with open(raw_path, "wb") as f:
        f.write(base64.b64decode(b64_audio))

    wav_path = tempfile.mkstemp(suffix=".wav")[1]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # just resample/mono; no filters
    cmd = [ffmpeg, "-y", "-i", raw_path, "-ac", "1", "-ar", "16000", "-f", "wav", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    rate, data = wavfile.read(wav_path)
    if data.size == 0:
        raise ValueError("Empty audio after conversion")

    if data.dtype != np.float32:
        data = data.astype(np.float32)
        m = max(1e-9, float(np.max(np.abs(data))))
        data = data / m

    # light trim of start/end silence
    data = _trim_silence(data, rate, frame_ms=30, thresh_db=-35.0)
    if data.size == 0:
        # fallback: don’t return empty, keep tiny slice
        data = np.zeros(1600, dtype=np.float32)

    return {"array": data, "sampling_rate": rate}
