# train.py
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------ CONFIG ------------------
DATA_ROOT = Path("data")
IMG_SIZE = (128, 128)   # smaller is faster to train; 224 works too
BATCH_SIZE = 32
EPOCHS = 12
MODEL_PATH = "mask_model.keras"
LABELS_JSON = "labels.json"
SEED = 42
# -------------------------------------------


def detect_layout(root: Path):
    """Return (train_dir, val_dir, use_internal_split) based on dataset layout."""
    candidates = [d.name.lower() for d in root.iterdir() if d.is_dir()]

    if "train" in candidates and "test" in candidates:
        # Layout B: pre-split dataset
        train_dir = root / ("train" if (root / "train").exists() else "Train")
        val_dir = root / ("test" if (root / "test").exists() else "Test")
        return train_dir, val_dir, False

    # Layout A: single folder, we’ll split internally
    return root, None, True


def build_datasets(train_dir: Path, val_dir, use_internal_split: bool):
    AUTOTUNE = tf.data.AUTOTUNE

    if use_internal_split:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            validation_split=0.2,
            subset="training",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="int",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )

    class_names = train_ds.class_names

    # Normalize
    normalization = tf.keras.Sequential([layers.Rescaling(1.0 / 255)])

    def norm_map(x, y):
        return normalization(x), y

    train_ds = train_ds.map(norm_map).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(norm_map).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(input_shape, num_classes):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax",
        ),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"Dataset folder not found: {DATA_ROOT.resolve()}")

    train_dir, val_dir, use_internal_split = detect_layout(DATA_ROOT)

    print("Preparing datasets…")
    train_ds, val_ds, class_names = build_datasets(train_dir, val_dir, use_internal_split)
    num_classes = len(class_names)
    print("Classes:", class_names)

    print("Building model…")
    model = build_model(IMG_SIZE + (3,), num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy", mode="max"),
    ]

    print("Training…")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print("Evaluating…")
    loss, acc = model.evaluate(val_ds)
    print(f"Validation accuracy: {acc:.3f}")

    # Save final model
    model.save(MODEL_PATH)

    # Save label mapping
    with open(LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "img_size": IMG_SIZE}, f, indent=2)

    print(f"Saved model to {MODEL_PATH} and labels to {LABELS_JSON}")


if __name__ == "__main__":
    main()
