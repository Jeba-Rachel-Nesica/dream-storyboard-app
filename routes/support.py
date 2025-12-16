from flask import Blueprint, jsonify
bp = Blueprint("support", __name__, url_prefix="/api/support")

@bp.get("/crisis")
def crisis():
    return jsonify({
        "notice":"If you feel unsafe, tap a crisis option below.",
        "phone":[{"label":"India: 112 (Emergency)","value":"112"},
                 {"label":"US: 988 Suicide & Crisis Lifeline","value":"988"}],
        "chat":[{"label":"Befrienders Worldwide","url":"https://www.befrienders.org/"}]
    }), 200
