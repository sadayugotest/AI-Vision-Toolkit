# -*- coding: utf-8 -*-
"""Worker: Keras Classification training."""

import os
import time

from ..config import RUNS_DIR
from ..models import TrainRequest, JobStatus
from ..state import CANCEL_REQUESTED
from ..utils import _update_time_stats, zip_artifacts


def run_keras_train(job_id: str, req: TrainRequest, job: JobStatus):
    """Run Keras classification training. Returns None (no yaml to clean)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as kcb

    dataset_root = os.path.abspath(req.dataset_root)

    # --- ตั้งค่า ---
    keras_model_name = req.keras_model or "MobileNetV2"
    lr = req.keras_lr or 0.0001
    freeze = req.keras_freeze if req.keras_freeze is not None else True
    img_size = req.imgsz if req.imgsz and req.imgsz > 0 else 224
    batch_size = req.batch if req.batch and req.batch > 0 else 32
    n_epochs = req.epochs if req.epochs and req.epochs > 0 else 30
    job.epochs = n_epochs

    results_base = os.path.join(RUNS_DIR, "classify", req.project_name + "_keras")
    os.makedirs(results_base, exist_ok=True)

    # --- Dataset (ImageFolder) ---
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"ไม่พบ {train_dir}")

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )
    class_names_found = train_ds.class_names
    num_classes = len(class_names_found)
    job.message = f"พบ {num_classes} classes: {', '.join(class_names_found[:10])}"

    val_ds = None
    if os.path.isdir(val_dir):
        val_ds = keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=False,
        )

    # Prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    if val_ds:
        val_ds = val_ds.prefetch(AUTOTUNE)

    # --- Base model (from scratch, ไม่ใช้ pretrained weights) ---
    base_models = {
        "MobileNetV2": keras.applications.MobileNetV2,
        "ResNet50": keras.applications.ResNet50,
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "InceptionV3": keras.applications.InceptionV3,
        "DenseNet121": keras.applications.DenseNet121,
    }
    BaseModelClass = base_models.get(keras_model_name, keras.applications.MobileNetV2)
    w_mode = None
    job.message = f"Keras {keras_model_name} — train from scratch"

    base_model = BaseModelClass(
        weights=w_mode,
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = not freeze

    # --- Build model ---
    inputs = keras.Input(shape=(img_size, img_size, 3))
    # Preprocessing: ใช้ Rescaling layer แทน Lambda เพื่อให้ .h5 โหลดกลับได้
    # scale [0,255] → [-1,1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    job.message = f"Keras {keras_model_name} (freeze={freeze}, lr={lr}, img={img_size})"

    # --- Callbacks ---
    best_path_keras = os.path.join(results_base, "best_model.keras")

    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if CANCEL_REQUESTED.get(job_id):
                job.state = "canceled"
                job.message = "ถูกยกเลิกโดยผู้ใช้"
                job.finished_at = time.time()
                self.model.stop_training = True
                return
            ep = epoch + 1
            job.epoch = ep
            job.percent = min(100.0, (ep / n_epochs) * 100.0)
            acc = logs.get("val_accuracy") or logs.get("accuracy", 0)
            job.map5095 = float(acc)
            _update_time_stats(job, time.time(), ep)
            job.message = (
                f"Epoch {ep}/{n_epochs}  "
                f"acc={logs.get('accuracy', 0):.4f}  "
                f"val_acc={logs.get('val_accuracy', 0):.4f}"
            )

    cb_list = [
        ProgressCallback(),
        kcb.ModelCheckpoint(
            best_path_keras,
            monitor="val_accuracy" if val_ds else "accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        kcb.EarlyStopping(
            monitor="val_accuracy" if val_ds else "accuracy",
            patience=max(10, n_epochs // 5),
            restore_best_weights=True,
            verbose=0,
        ),
        kcb.ReduceLROnPlateau(
            monitor="val_loss" if val_ds else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0,
        ),
    ]

    # --- Train ---
    job.percent = 1.0
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=cb_list,
        verbose=0,
    )
    if job.state != "canceled":
        job.percent = 100.0

    # --- Save final model ---
    final_path_keras = os.path.join(results_base, "final_model.keras")
    final_path_h5 = os.path.join(results_base, "final_model.h5")
    best_path_h5 = os.path.join(results_base, "best_model.h5")
    model.save(final_path_keras)
    model.save(final_path_h5, save_format="h5")

    # Convert best_model.keras → best_model.h5
    if os.path.exists(best_path_keras):
        try:
            best_loaded = keras.models.load_model(best_path_keras)
            best_loaded.save(best_path_h5, save_format="h5")
        except Exception:
            pass

    # Save class names
    cls_path = os.path.join(results_base, "class_names.txt")
    with open(cls_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names_found))

    # Save training summary
    summary_path = os.path.join(results_base, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Framework: Keras (TensorFlow)\n")
        f.write(f"Base Model: {keras_model_name}\n")
        f.write(f"Freeze: {freeze}\n")
        f.write(f"Image Size: {img_size}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {n_epochs}\n")
        f.write(f"Classes ({num_classes}): {', '.join(class_names_found)}\n")

    job.results_dir = results_base
    job.best_exists = os.path.exists(best_path_h5) or os.path.exists(best_path_keras)
    job.best_ckpt_path = best_path_h5 if os.path.exists(best_path_h5) else best_path_keras
    job.artifact_path = zip_artifacts(results_base) if results_base else None

    return None  # no yaml to clean
