from flask import Blueprint, request, jsonify, session
from pydantic import ValidationError
from sqlalchemy import select, desc
from database import SessionLocal
from models import Script
from schemas import RewriteIn
from services.model_service import generate_rewrite
from auth import login_required

bp = Blueprint("script", __name__, url_prefix="/api/script")

@bp.post("")
@login_required
def save_script():
    try:
        data = RewriteIn.model_validate_json(request.data)
    except ValidationError as e:
        return jsonify({"error":"invalid input","details":e.errors()}), 400
    uid = session["user_id"]
    rewrite = request.json.get("rewrite") or generate_rewrite(
        data.nightmare, data.controls.model_dump())["text"]
    with SessionLocal() as s, s.begin():
        rec = Script(user_id=uid, nightmare=data.nightmare,
                     controls=data.controls.model_dump(), rewrite=rewrite)
        s.add(rec); s.flush()
        return jsonify({"id": rec.id}), 201

@bp.get("")
@login_required
def list_scripts():
    uid = session["user_id"]
    with SessionLocal() as s:
        rows = s.execute(select(Script).where(Script.user_id==uid)
                         .order_by(desc(Script.created_at))).scalars().all()
        out = [{"id": r.id, "nightmare": r.nightmare, "rewrite": r.rewrite} for r in rows]
        return jsonify(out), 200
