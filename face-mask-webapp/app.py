import io
import json
import base64
from pathlib import Path
import os

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from pymongo import MongoClient
from dotenv import load_dotenv
from flask_login import LoginManager, login_required, current_user
from bson.objectid import ObjectId

# ML imports
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# load environment
load_dotenv()

# ---------- Config ----------
MODEL_PATH = "mask_model.keras"
LABELS_JSON = "labels.json"
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE_DEFAULT = (128, 128)

# MongoDB settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "facerecognition")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
# ----------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = SECRET_KEY

# ---------- MongoDB ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]
face_col = db["face-mask"]

users_col.create_index("email", unique=True)
app.db = db

# ---------- Flask-Login ----------
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

class SimpleUser:
    def __init__(self, user_doc):
        self.id = str(user_doc["_id"])
        self.email = user_doc.get("email")
        self.name = user_doc.get("name", "")

    def is_authenticated(self): return True
    def is_active(self): return True
    def is_anonymous(self): return False
    def get_id(self): return self.id

@login_manager.user_loader
def load_user(user_id):
    try:
        doc = users_col.find_one({"_id": ObjectId(user_id)})
        if not doc:
            return None
        return SimpleUser(doc)
    except Exception:
        return None

# ----- Auth blueprint -----
from auth import auth_bp
app.register_blueprint(auth_bp, url_prefix="")

# ---------- Load model ----------
if not Path(MODEL_PATH).exists():
    raise SystemExit(f"Model file not found at {MODEL_PATH}. Run train.py first.")

model = tf.keras.models.load_model(MODEL_PATH)
if Path(LABELS_JSON).exists():
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta.get("class_names", ["with_mask", "without_mask"])
    IMG_SIZE = tuple(meta.get("img_size", IMG_SIZE_DEFAULT))
else:
    class_names = ["with_mask", "without_mask"]
    IMG_SIZE = IMG_SIZE_DEFAULT

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
if face_cascade.empty():
    raise SystemExit("Failed to load Haar cascade. Check OpenCV installation.")

# ----------------- Helpers -----------------
def read_b64_to_cv2_img(b64str: str):
    header, _, data = b64str.partition(",")
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return arr

def predict_face(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    face_norm = face_resized.astype(np.float32) / 255.0
    face_input = np.expand_dims(face_norm, axis=0)
    prob = float(model.predict(face_input, verbose=0)[0][0])
    pred_idx = int(prob > 0.5)
    label = class_names[pred_idx]
    conf = prob if pred_idx == 1 else (1.0 - prob)
    return label, float(conf), pred_idx

# -------------- Routes ----------------
@app.route("/")
def index_public():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("index_public.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=getattr(current_user, "name", current_user.email))

@app.route("/predict_frame", methods=["POST"])
@login_required
def predict_frame():
    try:
        payload = request.get_json(force=True)
        img_b64 = payload.get("image", "")
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        frame = read_b64_to_cv2_img(img_b64)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        detections = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            label, conf, pred_idx = predict_face(face_roi)
            detections.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "label": label,
                "conf": round(float(conf), 4),
                "mask": ("mask" in label.lower())
            })

        return jsonify({"detections": detections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory("static", p)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
