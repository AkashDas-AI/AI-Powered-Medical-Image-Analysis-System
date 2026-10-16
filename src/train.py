"""Training orchestrator for the MobileNetV2 pneumonia classifier.

Provides the end-to-end training pipeline including data pipeline construction,
callback management, metrics tracking, and checkpoint serialization.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    PLOTS_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RSNA_BEST_MODEL_PATH,
    RSNA_DICOM_DIR,
    RSNA_FINAL_MODEL_PATH,
    RSNA_TEST_MANIFEST,
    RSNA_TRAIN_MANIFEST,
    RSNA_VAL_MANIFEST,
    SMOKE_TEST_MODEL_PATH,
)
from src.data_audit import audit_dataset
from src.model import build_mobilenet_v2
from src.rsna_dataset import RSNABatchGenerator

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    TENSORFLOW_AVAILABLE = False


def train_model(
    data_dir: Optional[Union[str, Path]] = None,
    model_save_path: Optional[Union[str, Path]] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weights: Optional[str] = "imagenet",
) -> Dict[str, Any]:
    """Execute the MobileNetV2 model training workflow.

    Workflow:
    1. Validates presence of dataset splits via data audit.
    2. Builds optimized tf.data.Dataset pipelines with MobileNetV2 scaling.
    3. Builds and compiles MobileNetV2 with transfer learning backbone.
    4. Configures EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and CSVLogger.
    5. Executes model training and saves both best and final checkpoints.

    Args:
        data_dir: Root dataset path containing train and val splits.
        model_save_path: Destination path for final model.
        epochs: Maximum number of training epochs. Defaults to config.
        batch_size: Batch size for gradient updates. Defaults to config.
        learning_rate: Learning rate for Adam optimizer. Defaults to config.
        weights: Pretrained backbone weights ('imagenet' or None).

    Returns:
        Dict[str, Any]: Summary dictionary of training execution and metrics.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If the dataset is not present or splits are missing.
    """
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError(
            "TensorFlow is required for model training. "
            "Install dependencies via pip install -r requirements.txt"
        )

    resolved_data_dir = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
    target_save_path = Path(model_save_path) if model_save_path is not None else DEFAULT_MODEL_PATH

    # 1. Audit dataset availability
    audit = audit_dataset(resolved_data_dir)
    if not audit["is_ready_for_training"]:
        raise FileNotFoundError(
            f"Dataset at '{resolved_data_dir}' is not ready for training. "
            f"Findings: {'; '.join(audit['warnings'])}. "
            "See data/README.md for download instructions."
        )

    resolved_root = Path(audit["resolved_data_root"])
    train_dir = resolved_root / "train"
    val_dir = resolved_root / "val"

    # 2. Construct tf.data pipelines
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    # Apply MobileNetV2 scaling: [-1.0, 1.0]
    normalization_layer = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    # 3. Build Model
    model = build_mobilenet_v2(
        num_classes=1,
        learning_rate=learning_rate,
        weights=weights,
        freeze_base=True,
    )

    # 4. Setup Callbacks
    target_save_path.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_log_path = REPORTS_DIR / "training_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path)),
    ]

    # 5. Execute Training Loop
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # 6. Save Final Checkpoint
    model.save(str(target_save_path))

    return {
        "status": "success",
        "epochs_completed": len(history.history["loss"]),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [-1.0])[-1]),
        "model_path": str(target_save_path),
        "best_model_path": str(BEST_MODEL_PATH),
        "history_csv": str(csv_log_path),
    }


def plot_training_curves(
    history_dict: Dict[str, List[float]],
    output_path: Path | str,
) -> Path:
    """Render and save training loss, accuracy, and AUROC convergence curves."""
    import matplotlib.pyplot as plt

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss
    axes[0].plot(epochs, history_dict.get("loss", []), "o-", label="Train Loss", color="#1f77b4")
    if "val_loss" in history_dict:
        axes[0].plot(epochs, history_dict["val_loss"], "s--", label="Val Loss", color="#d62728")
    axes[0].set_title("Binary Crossentropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Accuracy
    axes[1].plot(epochs, history_dict.get("accuracy", []), "o-", label="Train Acc", color="#2ca02c")
    if "val_accuracy" in history_dict:
        axes[1].plot(epochs, history_dict["val_accuracy"], "s--", label="Val Acc", color="#ff7f0e")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # 3. AUC
    axes[2].plot(epochs, history_dict.get("auc", []), "o-", label="Train AUC", color="#9467bd")
    if "val_auc" in history_dict:
        axes[2].plot(epochs, history_dict["val_auc"], "s--", label="Val AUC", color="#8c564b")
    axes[2].set_title("Area Under ROC (AUROC)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.suptitle("MobileNetV2 RSNA Transfer Learning Training Progression", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_path


def run_rsna_smoke_test(
    train_manifest: Path | str = RSNA_TRAIN_MANIFEST,
    dicom_root: Path | str = RSNA_DICOM_DIR,
    weights: Optional[str] = "imagenet",
    batch_size: int = 4,
    checkpoint_path: Path | str = SMOKE_TEST_MODEL_PATH,
) -> Dict[str, Any]:
    """Execute a comprehensive smoke test verifying architecture, batch streaming, and gradient updates.

    Verification steps:
    1. Instantiates RSNABatchGenerator and loads a small batch from disk.
    2. Builds MobileNetV2 architecture with specified weights.
    3. Confirms input/output tensor contracts.
    4. Executes forward pass and verifies valid probability bounds in [0.0, 1.0].
    5. Executes one backpropagation step with train_on_batch.
    6. Verifies finite loss and valid metrics.
    7. Saves a distinct smoke-test checkpoint.
    """
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError("TensorFlow is required for smoke testing.")

    manifest_p = Path(train_manifest)
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_p}")

    root_p = Path(dicom_root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {root_p}")

    # 1. Instantiate generator
    gen = RSNABatchGenerator(
        manifest_csv=manifest_p,
        dicom_root=root_p,
        batch_size=batch_size,
        shuffle=False,
    )
    if len(gen) == 0:
        raise ValueError("Training manifest has 0 records.")

    # 2. Build model
    model = build_mobilenet_v2(
        weights=weights,
        num_classes=1,
        freeze_base=True,
    )
    input_shape = model.input_shape
    output_shape = model.output_shape

    if input_shape != (None, 224, 224, 3):
        raise AssertionError(f"Unexpected input shape: {input_shape}")
    if output_shape != (None, 1):
        raise AssertionError(f"Unexpected output shape: {output_shape}")

    # 3. Process small batch
    batch_x, batch_y = gen[0]
    if batch_x.shape != (batch_size, 224, 224, 3):
        raise AssertionError(f"Batch X shape mismatch: {batch_x.shape}")
    if batch_y.shape != (batch_size,):
        raise AssertionError(f"Batch Y shape mismatch: {batch_y.shape}")
    import numpy as np
    if not np.isfinite(batch_x).all():
        raise AssertionError("Batch X contains non-finite values (NaN / Inf).")
    if batch_x.min() < -1.0 or batch_x.max() > 1.0:
        raise AssertionError(f"Batch X values exceed [-1.0, 1.0]: min={batch_x.min()}, max={batch_x.max()}")

    # 4. Forward pass
    fwd_out = model(batch_x, training=False)
    fwd_arr = fwd_out.numpy()
    if fwd_arr.shape != (batch_size, 1):
        raise AssertionError(f"Forward output shape mismatch: {fwd_arr.shape}")
    if not np.isfinite(fwd_arr).all():
        raise AssertionError("Forward output contains non-finite values.")
    if fwd_arr.min() < 0.0 or fwd_arr.max() > 1.0:
        raise AssertionError(f"Sigmoid outputs exceed [0.0, 1.0]: min={fwd_arr.min()}, max={fwd_arr.max()}")

    # 5. Backward pass / training step
    step_metrics = model.train_on_batch(batch_x, batch_y)
    loss_val = float(step_metrics[0]) if isinstance(step_metrics, (list, tuple)) else float(step_metrics)
    if not np.isfinite(loss_val) or loss_val <= 0:
        raise AssertionError(f"Training step loss invalid: {loss_val}")

    # 6. Save smoke-test checkpoint
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_path))

    return {
        "status": "PASS",
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "batch_size_tested": batch_size,
        "batch_x_shape": list(batch_x.shape),
        "batch_y_shape": list(batch_y.shape),
        "batch_y_labels": [int(v) for v in batch_y],
        "forward_min": float(fwd_arr.min()),
        "forward_max": float(fwd_arr.max()),
        "loss_after_one_step": loss_val,
        "step_metrics": [float(m) for m in step_metrics] if isinstance(step_metrics, (list, tuple)) else [loss_val],
        "smoke_checkpoint_path": str(ckpt_path),
    }


def train_rsna_baseline(
    train_manifest: Path | str = RSNA_TRAIN_MANIFEST,
    val_manifest: Path | str = RSNA_VAL_MANIFEST,
    dicom_root: Path | str = RSNA_DICOM_DIR,
    model_save_path: Path | str = RSNA_FINAL_MODEL_PATH,
    best_model_path: Path | str = RSNA_BEST_MODEL_PATH,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    freeze_base: bool = True,
    weights: Optional[str] = "imagenet",
    steps_per_epoch: Optional[int] = None,
    val_steps: Optional[int] = None,
    output_dir: Path | str = REPORTS_DIR,
    plots_dir: Path | str = PLOTS_DIR,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute end-to-end MobileNetV2 transfer-learning training on the RSNA dataset.

    Features:
    - Bounded memory streaming via RSNABatchGenerator.
    - Balanced class weighting calculated strictly from train split distribution.
    - Early stopping and model checkpointing on validation loss.
    - Learning rate annealing via ReduceLROnPlateau.
    - Full training metrics logging (CSV and JSON) and convergence plotting.
    - Zero data leakage: test manifest is completely unreferenced.
    """
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError("TensorFlow is required for model training.")

    train_p = Path(train_manifest)
    val_p = Path(val_manifest)
    dicom_p = Path(dicom_root)
    save_final_p = Path(model_save_path)
    save_best_p = Path(best_model_path)
    out_dir = Path(output_dir)
    plot_dir = Path(plots_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_final_p.parent.mkdir(parents=True, exist_ok=True)
    save_best_p.parent.mkdir(parents=True, exist_ok=True)

    if not train_p.is_file():
        raise FileNotFoundError(f"Train manifest not found: {train_p}")
    if not val_p.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {val_p}")

    # Set seeds
    tf.random.set_seed(seed)
    import numpy as np
    np.random.seed(seed)

    # 1. Instantiate Generators
    train_gen = RSNABatchGenerator(
        manifest_csv=train_p,
        dicom_root=dicom_p,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_gen = RSNABatchGenerator(
        manifest_csv=val_p,
        dicom_root=dicom_p,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    # 2. Derive Class Weights Exclusively from Training Split
    # Formula: w_j = N / (K * N_j)
    class_weights = train_gen.compute_balanced_class_weights()
    train_counts = train_gen.class_counts

    logger.info(f"[TRAIN-RSNA] Training samples: {train_gen.total_samples}, Val samples: {val_gen.total_samples}")
    logger.info(f"[TRAIN-RSNA] Class counts in train: {train_counts}")
    logger.info(f"[TRAIN-RSNA] Derived balanced class weights: {class_weights}")

    # 3. Build MobileNetV2 Model
    model = build_mobilenet_v2(
        num_classes=1,
        learning_rate=learning_rate,
        freeze_base=freeze_base,
        weights=weights,
    )

    # 4. Configure Callbacks
    csv_log_path = out_dir / "rsna_training_history.csv"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_best_p),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path)),
    ]

    # Compute effective steps and verify data coverage
    total_train_batches = len(train_gen)
    total_val_batches = len(val_gen)
    eff_train_steps = min(total_train_batches, steps_per_epoch) if steps_per_epoch else total_train_batches
    eff_val_steps = min(total_val_batches, val_steps) if val_steps else total_val_batches

    is_full_training_pass = bool(steps_per_epoch is None or steps_per_epoch >= total_train_batches)
    is_full_validation_pass = bool(val_steps is None or val_steps >= total_val_batches)
    train_images_per_epoch = train_gen.total_samples if is_full_training_pass else eff_train_steps * batch_size
    val_images_per_epoch = val_gen.total_samples if is_full_validation_pass else eff_val_steps * batch_size
    train_coverage_pct = round(min(100.0, float(train_images_per_epoch / train_gen.total_samples * 100.0)), 2)
    val_coverage_pct = round(min(100.0, float(val_images_per_epoch / val_gen.total_samples * 100.0)), 2)

    experiment_type = (
        "FULL_PASS_EXPERIMENT"
        if (is_full_training_pass and is_full_validation_pass)
        else "STEP_LIMITED_PILOT"
    )

    logger.info(
        f"[TRAIN-RSNA] Experiment Type: {experiment_type} | "
        f"Train Coverage: {train_coverage_pct}% ({train_images_per_epoch:,} / {train_gen.total_samples:,} imgs, {eff_train_steps} batches) | "
        f"Val Coverage: {val_coverage_pct}% ({val_images_per_epoch:,} / {val_gen.total_samples:,} imgs, {eff_val_steps} batches)"
    )

    # Record training configuration metadata
    config_record = {
        "experiment_type": experiment_type,
        "is_full_training_pass": is_full_training_pass,
        "is_full_validation_pass": is_full_validation_pass,
        "train_coverage_percentage": train_coverage_pct,
        "val_coverage_percentage": val_coverage_pct,
        "train_images_per_epoch": train_images_per_epoch,
        "val_images_per_epoch": val_images_per_epoch,
        "model_architecture": "MobileNetV2",
        "output_layer_name": "lung_opacity_prob",
        "input_shape": [224, 224, 3],
        "pretrained_weights": weights,
        "freeze_backbone": freeze_base,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "train_samples": train_gen.total_samples,
        "val_samples": val_gen.total_samples,
        "train_steps_per_epoch": eff_train_steps,
        "val_steps_per_epoch": eff_val_steps,
        "class_counts_train": train_counts,
        "class_weights": class_weights,
        "class_weight_formula": "w_j = N_train / (2 * N_j)",
        "train_manifest": str(train_p),
        "val_manifest": str(val_p),
        "random_seed": seed,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_path = out_dir / "rsna_training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    # 5. Execute Training Loop
    t_start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=eff_train_steps,
        validation_steps=eff_val_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    duration_sec = time.time() - t_start

    # 6. Save Final Checkpoint
    model.save(str(save_final_p))

    # 7. Generate Training Curve Plots
    plot_file = plot_dir / "rsna_training_history.png"
    try:
        plot_training_curves(history.history, plot_file)
    except Exception as exc:
        logger.warning(f"Could not generate training plot: {exc}")

    return {
        "status": "success",
        "experiment_type": experiment_type,
        "is_full_training_pass": is_full_training_pass,
        "is_full_validation_pass": is_full_validation_pass,
        "train_coverage_percentage": train_coverage_pct,
        "val_coverage_percentage": val_coverage_pct,
        "train_images_per_epoch": train_images_per_epoch,
        "val_images_per_epoch": val_images_per_epoch,
        "epochs_completed": len(history.history["loss"]),
        "training_duration_seconds": round(duration_sec, 2),
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", [-1.0])[-1]),
        "final_auc": float(history.history.get("auc", [-1.0])[-1]),
        "final_val_auc": float(history.history.get("val_auc", [-1.0])[-1]),
        "best_model_path": str(save_best_p),
        "final_model_path": str(save_final_p),
        "history_csv": str(csv_log_path),
        "config_json": str(config_path),
        "plot_path": str(plot_file),
    }

