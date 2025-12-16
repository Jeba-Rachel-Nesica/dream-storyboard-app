import os
from flask import Blueprint, request, jsonify, send_from_directory
from pydantic import ValidationError
from schemas import TTSIn
from services.tts_service import synth_to_file
from config import Config
from auth import login_required

bp = Blueprint("tts", __name__, url_prefix="/api")

@bp.post("/tts")
@login_required
def tts():
    try:
        data = TTSIn.model_validate_json(request.data)
    except ValidationError as e:
        return jsonify({"error":"invalid input","details":e.errors()}), 400
    path = synth_to_file(data.text)
    return jsonify({"url": f"/media/{os.path.basename(path)}"}), 200

media_bp = Blueprint("media", __name__)

@media_bp.get("/media/<fname>")
def serve_media(fname):
    return send_from_directory(Config.MEDIA_DIR, fname, as_attachment=False)
