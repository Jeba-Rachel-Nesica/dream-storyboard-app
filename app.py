from datetime import datetime
import os, base64, tempfile, subprocess
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required, logout_user, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash

import numpy as np
from scipy.io import wavfile
import imageio_ffmpeg

# ===== ML pieces (your PPO comfort model + ASR + TTS) =====
from generate_script import load_model, generate_batch
from services.audio_utils import ffmpeg_to_wav_array_from_b64
from services.asr_utils import run_asr_from_array, preload_asr
from services.tts_utils import tts_base64

# -------------------- Flask & DB setup --------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.update(
    SECRET_KEY="safe-grounded-app",
    SQLALCHEMY_DATABASE_URI="sqlite:///app.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)
CORS(app, supports_credentials=True)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# -------------------- User media storage --------------------
MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "user_data")
os.makedirs(MEDIA_ROOT, exist_ok=True)

def user_media_dir(uid: int) -> str:
    d = os.path.join(MEDIA_ROOT, str(uid))
    os.makedirs(d, exist_ok=True)
    return d

# -------------------- DB MODELS --------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    name = db.Column(db.String(120), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scripts = db.relationship("Script", backref="user", lazy=True, cascade="all,delete")
    journals = db.relationship("Journal", backref="user", lazy=True, cascade="all,delete")
    rehearsals = db.relationship("Rehearsal", backref="user", lazy=True, cascade="all,delete")

class Script(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    nightmare = db.Column(db.Text, nullable=False)
    comfort = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Journal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    sleep_quality = db.Column(db.Integer)
    nightmare_intensity = db.Column(db.Integer)
    emotions = db.Column(db.Text)
    note = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Rehearsal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(200), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)   # on-disk WAV path
    duration_s = db.Column(db.Float)                        # seconds
    script_text = db.Column(db.Text, nullable=False)        # full comfort script used
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))

with app.app_context():
    db.create_all()

# -------------------- Load Models --------------------
print("🔧 Loading comfort model + Whisper…")
tok, model, device = load_model("checkpoints/ppo_comfort/final_model")
model.eval()
print(f"[load] device={device}")
preload_asr()  # warm ASR to avoid first-call stall

# -------------------- Pages --------------------
@app.get("/")
def home(): return render_template("home.html")

@app.get("/consent")
def consent(): return render_template("consent.html")

@app.get("/login")
def login(): return render_template("login.html")

@app.post("/login")
def login_post():
    data = request.form or request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        flash("Invalid credentials.", "error")
        return redirect(url_for("login"))
    login_user(user)
    return redirect(url_for("write"))

@app.get("/signup")
def signup(): return render_template("signup.html")

@app.post("/signup")
def signup_post():
    data = request.form or request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    password = data.get("password") or ""
    if not email or not password:
        flash("Email and password required.", "error"); return redirect(url_for("signup"))
    if User.query.filter_by(email=email).first():
        flash("Account already exists.", "error"); return redirect(url_for("signup"))
    user = User(email=email, name=name, password_hash=generate_password_hash(password))
    db.session.add(user); db.session.commit()
    login_user(user)
    return redirect(url_for("write"))

@app.post("/logout")
@login_required
def logout(): logout_user(); return redirect(url_for("home"))

@app.get("/write")
@login_required
def write(): return render_template("write.html")

@app.get("/rehearse")
@login_required
def rehearse():
    last = Script.query.filter_by(user_id=current_user.id).order_by(Script.created_at.desc()).first()
    return render_template("rehearse.html", last_script=(last.comfort if last else ""))

@app.get("/rehearsals")
@login_required
def rehearsals_page():
    return render_template("rehearsals.html")

@app.get("/journal")
@login_required
def journal(): return render_template("journal.html")

@app.get("/support")
def support(): return render_template("support.html")

# -------------------- API --------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "user": current_user.get_id() if current_user.is_authenticated else None})

@app.post("/rewrite")
@login_required
def rewrite():
    """
    Accepts:
      { mode:"text", nightmare:"..." }
      or { mode:"voice", audio_base64:"...", audio_mime:"audio/webm|audio/ogg" }
    Returns:
      { asr_text, comfort_text, comfort_audio_base64, comfort_audio_mime }
    """
    try:
        data = request.get_json(force=True) or {}
        mode = (data.get("mode") or "text").lower()

        if mode == "voice":
            audio_b64 = data.get("audio_base64")
            if not audio_b64:
                return jsonify({"error": "audio_base64 required for voice mode"}), 400
            mime = data.get("audio_mime", "audio/webm")
            audio = ffmpeg_to_wav_array_from_b64(audio_b64, mime)
            # No prompt bias
            nightmare_text = run_asr_from_array(audio["array"], audio["sampling_rate"], max_seconds=20)
            if not nightmare_text:
                return jsonify({"error": "No speech detected. Try again closer to the mic."}), 400
            asr_text = nightmare_text
        else:
            nightmare_text = (data.get("nightmare") or "").strip()
            if not nightmare_text:
                return jsonify({"error": "Please describe your dream."}), 400
            asr_text = nightmare_text

        comforts = generate_batch(tok, model, device, [nightmare_text])
        comfort_text = comforts[0] if comforts else ""

        rec = Script(user_id=current_user.id, nightmare=nightmare_text, comfort=comfort_text)
        db.session.add(rec); db.session.commit()

        tts = tts_base64(comfort_text)  # {"b64": "...", "mime": "audio/mpeg|audio/wav"} or empty

        return jsonify({
            "asr_text": asr_text,
            "comfort_text": comfort_text,
            "comfort_audio_base64": tts.get("b64", ""),
            "comfort_audio_mime": tts.get("mime", "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/get_last_script")
@login_required
def get_last_script():
    last = Script.query.filter_by(user_id=current_user.id).order_by(Script.created_at.desc()).first()
    return jsonify({"script": last.comfort if last else ""})

@app.post("/save_journal")
@login_required
def save_journal():
    entry = request.get_json(force=True) or {}
    j = Journal(
        user_id=current_user.id,
        sleep_quality=entry.get("sleep_quality"),
        nightmare_intensity=entry.get("nightmare_intensity"),
        emotions=entry.get("emotions"),
        note=entry.get("note")
    )
    db.session.add(j); db.session.commit()
    return jsonify({"ok": True})

@app.get("/get_journal")
@login_required
def get_journal():
    rows = Journal.query.filter_by(user_id=current_user.id).order_by(Journal.created_at.desc()).all()
    return jsonify([
        {"sleep_quality": r.sleep_quality, "nightmare_intensity": r.nightmare_intensity,
         "emotions": r.emotions, "note": r.note, "created_at": r.created_at.isoformat()}
        for r in rows
    ])

@app.post("/tts")
@login_required
def tts_only():
    try:
        data = request.get_json(force=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        tts = tts_base64(text)
        return jsonify({"audio_base64": tts.get("b64", ""), "audio_mime": tts.get("mime", "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Rehearsal save/list/stream --------------------
# -------------------- Rehearsal save/list/stream (single source of truth) --------------------
@app.post("/rehearsal/save", endpoint="rehearsal_save_api")
@login_required
def rehearsal_save_api():
    """
    Accepts: { script_text?: string, title?: string, pause_ms?: int }
    Creates one WAV with short pauses. Only commits to DB if the file exists and has audio.
    """
    try:
        data = request.get_json(force=True) or {}
        title = (data.get("title") or "Rehearsal").strip()
        pause_ms = int(data.get("pause_ms") or 800)

        # Get script
        script = (data.get("script_text") or "").strip()
        if not script:
            last = Script.query.filter_by(user_id=current_user.id).order_by(Script.created_at.desc()).first()
            if not last or not (last.comfort or "").strip():
                return jsonify({"error": "No script available to save."}), 400
            script = last.comfort.strip()

        # Split into sentences
        import re, numpy as np, base64, tempfile, subprocess, os
        from scipy.io import wavfile
        import imageio_ffmpeg

        lines = [s for s in re.split(r'(?<=[.!?])\s+', script) if s]
        if not lines:
            return jsonify({"error": "No sentences to synthesize."}), 400

        sr = 16000
        segments = []
        made_any = False

        for s in lines:
            # TTS util returns {"b64": "...", "mime": "audio/mpeg|audio/wav"} or empty
            t = tts_base64(s)
            b64 = (t or {}).get("b64") or ""
            if not b64:
                continue

            made_any = True
            ext = ".mp3" if "mpeg" in ((t or {}).get("mime") or "") else ".wav"
            tmp_in = tempfile.mkstemp(suffix=ext)[1]
            with open(tmp_in, "wb") as f:
                f.write(base64.b64decode(b64))

            tmp_wav = tempfile.mkstemp(suffix=".wav")[1]
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg, "-y", "-i", tmp_in, "-ac", "1", "-ar", str(sr), "-f", "wav", tmp_wav]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

            rate, data_wav = wavfile.read(tmp_wav)
            if getattr(data_wav, "size", 0) == 0:
                continue
            if data_wav.dtype != np.float32:
                data_wav = data_wav.astype(np.float32)
                m = float(np.max(np.abs(data_wav))) or 1.0
                data_wav = data_wav / m
            segments.append(data_wav)

            # pause between lines
            pause = np.zeros(int(sr * (pause_ms/1000.0)), dtype=np.float32)
            segments.append(pause)

        if not (made_any and segments):
            return jsonify({"error": "Could not synthesize audio (no TTS output)."}), 500

        full = np.concatenate(segments)
        if not np.any(np.abs(full) > 1e-6):
            return jsonify({"error": "Audio was silent; nothing saved."}), 500

        duration_s = round(full.shape[0] / sr, 2)
        out_dir = user_media_dir(current_user.id)
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip() or "Rehearsal"
        filename = f"{safe_title}_{int(datetime.utcnow().timestamp())}.wav"
        out_path = os.path.join(out_dir, filename)
        wavfile.write(out_path, sr, (full * 32767.0).astype(np.int16))

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return jsonify({"error": "File write failed; nothing saved."}), 500

        rec = Rehearsal(
            user_id=current_user.id,
            title=title,
            filepath=out_path,
            duration_s=float(duration_s),
            script_text=script,
        )
        db.session.add(rec); db.session.commit()

        return jsonify({"id": rec.id, "title": rec.title, "url": f"/rehearsal/{rec.id}/audio"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/rehearsal/list", endpoint="rehearsal_list_api")
@login_required
def rehearsal_list_api():
    import os
    rows = Rehearsal.query.filter_by(user_id=current_user.id).order_by(Rehearsal.created_at.desc()).all()
    return jsonify([
        {
            "id": r.id,
            "title": r.title,
            "duration_s": r.duration_s,
            "created_at": r.created_at.isoformat(),
            "url": f"/rehearsal/{r.id}/audio",
            "has_file": bool(os.path.exists(r.filepath) and os.path.getsize(r.filepath) > 0),
        }
        for r in rows
    ])


@app.get("/rehearsal/<int:rid>/audio", endpoint="rehearsal_audio_api")
@login_required
def rehearsal_audio_api(rid):
    import os
    r = db.session.get(Rehearsal, rid)
    if not r or r.user_id != current_user.id:
        return jsonify({"error": "Not found"}), 404
    if not os.path.exists(r.filepath) or os.path.getsize(r.filepath) == 0:
        return jsonify({"error": "File missing"}), 404
    return send_file(
        r.filepath,
        mimetype="audio/wav",
        as_attachment=False,
        conditional=True,  # enables range requests for <audio>
        download_name=os.path.basename(r.filepath),
        max_age=0,
        etag=True,
        last_modified=datetime.utcfromtimestamp(os.path.getmtime(r.filepath)),
    )
