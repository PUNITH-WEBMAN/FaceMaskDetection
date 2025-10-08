# infer_labels.py
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

# Local paths (already copied model into webapp)
MODEL_PATH = "mask_model.keras"
DATA_ROOT = Path(r"C:\Users\Punith\OneDrive\Desktop\Ai_Datascience_project\FaceMask\data")
OUT_JSON = Path("labels.json")
IMG_SIZE = (128, 128)  # change to the size you trained with if different

# Load model
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

def find_sample(folder: Path):
    if not folder.exists():
        return None
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        files = list(folder.glob(ext))
        if files:
            return files[0]
    return None

with_mask_img = find_sample(DATA_ROOT / "with_mask")
without_mask_img = find_sample(DATA_ROOT / "without_mask")

if with_mask_img is None or without_mask_img is None:
    print("Could not find sample images. Check DATA_ROOT and that folders 'with_mask' and 'without_mask' exist.")
    raise SystemExit(1)

def preprocess(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

p_with = float(model.predict(preprocess(with_mask_img), verbose=0)[0][0])
p_without = float(model.predict(preprocess(without_mask_img), verbose=0)[0][0])

print("Prob(with_mask sample)   =", p_with)
print("Prob(without_mask sample)=", p_without)

# If model output larger means class index 1; choose ordering so class_names[index] matches label meaning.
# If p_with > p_without, index 1 likely corresponds to 'with_mask'
if p_with > p_without:
    class_names = ["without_mask", "with_mask"]
else:
    class_names = ["with_mask", "without_mask"]

OUT_JSON.write_text(json.dumps({"class_names": class_names, "img_size": list(IMG_SIZE)}, indent=2), encoding="utf-8")
print(f"Wrote {OUT_JSON} with class_names = {class_names}")
