from flask import Blueprint, request, jsonify, session
from pydantic import ValidationError
from sqlalchemy import select, desc
from database import SessionLocal
from models import JournalEntry
from schemas import JournalIn
from auth import login_required

bp = Blueprint("journal", __name__, url_prefix="/api/journal")

@bp.post("")
@login_required
def add_entry():
    try:
        data = JournalIn.model_validate_json(request.data)
    except ValidationError as e:
        return jsonify({"error":"invalid input","details":e.errors()}), 400
    uid = session["user_id"]
    with SessionLocal() as s, s.begin():
        j = JournalEntry(user_id=uid, sleep_quality=data.sleep_quality,
                         nightmare_intensity=data.nightmare_intensity,
                         emotions=(data.emotions or "")[:120],
                         note=(data.note or ""))
        s.add(j); s.flush()
        return jsonify({"id": j.id}), 201

@bp.get("")
@login_required
def list_entries():
    uid = session["user_id"]
    with SessionLocal() as s:
        rows = s.execute(select(JournalEntry).where(JournalEntry.user_id==uid)
                         .order_by(desc(JournalEntry.created_at))).scalars().all()
        out = [{
            "id": r.id, "sleep_quality": r.sleep_quality,
            "nightmare_intensity": r.nightmare_intensity,
            "emotions": r.emotions, "note": r.note,
            "created_at": r.created_at.isoformat()
        } for r in rows]
        return jsonify(out), 200
