# auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from models import db, User

auth_bp = Blueprint("auth", __name__)

@auth_bp.get("/login")
def login_get():
    return render_template("login.html")

@auth_bp.post("/login")
def login_post():
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        flash("Invalid email or password", "error")
        return redirect(url_for("auth.login_get"))
    login_user(user, remember=True)
    return redirect(url_for("home"))

@auth_bp.get("/register")
def register_get():
    return render_template("register.html")

@auth_bp.post("/register")
def register_post():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not name or not email or not password:
        flash("Please fill all fields", "error")
        return redirect(url_for("auth.register_get"))
    if User.query.filter_by(email=email).first():
        flash("Email already registered", "error")
        return redirect(url_for("auth.register_get"))
    user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password)
    )
    db.session.add(user)
    db.session.commit()
    login_user(user, remember=True)
    return redirect(url_for("home"))

@auth_bp.post("/logout")
@login_required
def logout_post():
    logout_user()
    return redirect(url_for("auth.login_get"))
