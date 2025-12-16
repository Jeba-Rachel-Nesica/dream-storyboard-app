from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from schemas import RewriteIn, RewriteOut
from services.model_service import generate_rewrite
from auth import login_required

bp = Blueprint("rewrite", __name__, url_prefix="/api")

@bp.post("/rewrite")
@login_required
def rewrite():
    try:
        data = RewriteIn.model_validate_json(request.data)
    except ValidationError as e:
        return jsonify({"error": "invalid input", "details": e.errors()}), 400
    result = generate_rewrite(data.nightmare, data.controls.model_dump())
    payload = RewriteOut(text=result["text"]).model_dump()
    payload["flags"] = result.get("flags", {})
    return jsonify(payload), 200
