# auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, UserMixin
from bson.objectid import ObjectId

auth_bp = Blueprint("auth", __name__, template_folder="templates")

# Simple User wrapper for Flask-Login
class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc["_id"])
        self.email = user_doc.get("email")
        self.name = user_doc.get("name", "")

# Helper to access users collection
def users_coll():
    return current_app.db["users"]

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email & password required", "warning")
            return redirect(url_for("auth.register"))

        # Check existing
        if users_coll().find_one({"email": email}):
            flash("Email already registered. Please login.", "danger")
            return redirect(url_for("auth.login"))

        hashed = generate_password_hash(password)
        user_doc = {"name": name, "email": email, "password": hashed}
        res = users_coll().insert_one(user_doc)
        user_doc["_id"] = res.inserted_id

        flash("Registration successful — please log in", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user_doc = users_coll().find_one({"email": email})
        if not user_doc:
            flash("Invalid credentials", "danger")
            return redirect(url_for("auth.login"))

        if not check_password_hash(user_doc["password"], password):
            flash("Invalid credentials", "danger")
            return redirect(url_for("auth.login"))

        user = User(user_doc)
        login_user(user)
        flash("Logged in successfully", "success")
        # redirect to next or dashboard
        next_page = request.args.get("next") or url_for("dashboard")
        return redirect(next_page)

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for("auth.login"))
