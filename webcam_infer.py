# webcam_infer.py
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def load_model_and_meta(model_path: Path, labels_json: Path):
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path.resolve()}")
    if not labels_json.exists():
        raise SystemExit(f"Labels JSON not found: {labels_json.resolve()}")

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))

    print(f"Loading label metadata from {labels_json} ...")
    with open(labels_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_names = meta.get("class_names")
    if not class_names or not isinstance(class_names, list):
        raise SystemExit("Invalid or missing 'class_names' in labels JSON.")

    img_size = tuple(int(x) for x in meta.get("img_size", [128, 128]))
    if len(img_size) != 2:
        raise SystemExit("Invalid 'img_size' in labels JSON; expected [H, W].")

    return model, class_names, img_size


def safe_open_camera(camera_index: int, width: int = 640, height: int = 480, tries: int = 3):
    # Try preferred index first, then fallback indices
    for idx in ([camera_index] + [i for i in range(0, tries) if i != camera_index]):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if cap is None or not cap.isOpened():
            print(f"Camera index {idx} not available, trying next...")
            continue
        # set desirable properties; these may be ignored by some webcams
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"Opened camera index {idx}.")
        return cap
    raise SystemExit("Could not open any webcam. Try a different camera index or check permissions.")


def make_prediction(model, face_rgb, img_size, threshold):
    """
    face_rgb: face patch in RGB uint8 (HxWx3)
    returns: pred_idx (int), conf (float), debug_info (dict)
    """
    # resize to model input size (preserve shape exactly as used during training)
    face_resized = cv2.resize(face_rgb, img_size)
    face_norm = face_resized.astype(np.float32) / 255.0
    face_input = np.expand_dims(face_norm, axis=0)  # shape (1, H, W, 3)

    preds = model.predict(face_input, verbose=0)
    preds = np.asarray(preds)

    # Interpret outputs robustly:
    # - binary sigmoid -> shape (1,1) or (1,)
    # - multi-class softmax -> shape (1, num_classes)
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0, 0])
        pred_idx = int(prob > threshold)
        conf = prob if pred_idx == 1 else 1.0 - prob
        debug = {"mode": "binary", "raw": prob}
        return pred_idx, float(conf), debug

    if preds.ndim == 2 and preds.shape[0] == 1:
        probs = preds[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        debug = {"mode": "multiclass", "raw": probs.tolist()}
        return pred_idx, float(conf), debug

    # fallback: try flatten
    flat = preds.flatten()
    if flat.size == 1:
        prob = float(flat[0])
        pred_idx = int(prob > threshold)
        conf = prob if pred_idx == 1 else 1.0 - prob
        debug = {"mode": "binary_flat", "raw": prob}
        return pred_idx, float(conf), debug

    # worst-case: choose argmax on flattened
    pred_idx = int(np.argmax(flat))
    conf = float(flat[pred_idx])
    debug = {"mode": "fallback_flat", "raw": flat.tolist()}
    return pred_idx, float(conf), debug


def choose_color_for_label(label: str, default_green_contains=("mask", "with")):
    low = label.lower()
    if any(k in low for k in default_green_contains):
        return (0, 255, 0)  # green
    if any(k in low for k in ("without", "no", "nomask", "no_mask", "nocover")):
        return (0, 0, 255)  # red
    return (255, 200, 0)  # amber/blue-ish for unknown labels


def main():
    parser = argparse.ArgumentParser(description="Webcam mask detection inference (robust).")
    parser.add_argument("--model", type=Path, default=Path("mask_model.keras"), help="Path to the Keras model (.keras)")
    parser.add_argument("--labels", type=Path, default=Path("labels.json"), help="Path to labels.json")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for sigmoid outputs")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument("--show-fps", action="store_true", help="Show FPS on the display")
    args = parser.parse_args()

    model, class_names, img_size = load_model_and_meta(args.model, args.labels)
    print("Class names (index → name):", list(enumerate(class_names)))
    print("Model input image size:", img_size)

    # Load Haar cascade; if it fails, warn and use whole-frame ROI as fallback
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Warning: Haar cascade failed to load. Face detection will be skipped; using full-frame as ROI.")
        face_cascade = None
    else:
        print(f"Loaded Haar cascade from {cascade_path}")

    cap = safe_open_camera(args.camera, width=args.width, height=args.height)

    print("Press 'q' to quit. (Ctrl+C also works)")
    last_time = time.time()
    fps_smooth = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: frame grab failed, stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = []
            if face_cascade is not None:
                # detectMultiScale returns (x,y,w,h) tuples
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            else:
                # fallback: treat whole frame as single face ROI
                h, w = frame.shape[:2]
                faces = [(0, 0, w, h)]

            for (x, y, w, h) in faces:
                # Extract & preprocess face ROI (convert BGR->RGB!)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                face_roi = frame[y1:y2, x1:x2]
                # convert to RGB (tf.keras expects RGB images when trained from image_dataset_from_directory)
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                try:
                    pred_idx, conf, debug = make_prediction(model, face_rgb, img_size, args.threshold)
                except Exception as e:
                    # Prediction failed for some reason; draw grey box and continue
                    print("Prediction error:", e)
                    pred_idx, conf = 0, 0.0
                    debug = {"mode": "error", "error": str(e)}

                label = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else f"idx{pred_idx}"
                color = choose_color_for_label(label)

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ({conf*100:.1f}%)"
                cv2.putText(frame, text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # FPS calculation
            if args.show_fps:
                now = time.time()
                dt = now - last_time
                last_time = now
                fps = 1.0 / dt if dt > 0 else 0.0
                # exponential smoothing for readability
                fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.8 + fps * 0.2)
                cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Mask Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
