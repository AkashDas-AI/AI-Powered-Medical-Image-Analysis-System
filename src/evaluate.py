"""Evaluation and clinical metrics benchmarking module.

Computes comprehensive diagnostic screening metrics (Sensitivity, Specificity,
AUROC, F1, Confusion Matrix) on test datasets and generates clinical summary artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    IMAGE_SIZE,
    PLOTS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RSNA_BEST_MODEL_PATH,
    RSNA_CLASS_NAMES,
    RSNA_DICOM_DIR,
    RSNA_FINAL_MODEL_PATH,
    RSNA_POSITIVE_CLASS,
    RSNA_TEST_MANIFEST,
)
from src.rsna_dataset import RSNABatchGenerator

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    TENSORFLOW_AVAILABLE = False


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Compute clinical diagnostic screening metrics from ground truth and predictions.

    Handles edge cases with undefined metrics (e.g., zero positive or negative samples,
    empty arrays, single-class cohorts) safely by reporting None with diagnostic notes.

    Args:
        y_true: 1D binary array of ground truth labels (0 = NORMAL, 1 = PNEUMONIA).
        y_prob: 1D float array of predicted pneumonia probabilities in [0.0, 1.0].
        threshold: Fixed decision threshold for binarizing predictions. Default: 0.5.

    Returns:
        Dict[str, Any]: Dictionary containing Sensitivity, Specificity, AUROC,
            Accuracy, Precision, F1, confusion matrix, and undefined metric notes.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    total_samples = int(len(y_true))
    if total_samples == 0:
        return {
            "threshold": float(threshold),
            "total_samples": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "sensitivity_recall": None,
            "specificity": None,
            "precision": None,
            "accuracy": None,
            "f1_score": None,
            "auroc": None,
            "notes": ["Empty evaluation cohort provided."],
        }

    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix elements: TP, FP, TN, FN
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Clinical screening metrics with undefined edge case handling
    notes = []
    if (tp + fn) > 0:
        sensitivity = round(float(tp / (tp + fn)), 4)
    else:
        sensitivity = None
        notes.append("Sensitivity undefined: cohort contains zero positive (PNEUMONIA) cases.")

    if (tn + fp) > 0:
        specificity = round(float(tn / (tn + fp)), 4)
    else:
        specificity = None
        notes.append("Specificity undefined: cohort contains zero negative (NORMAL) cases.")

    if (tp + fp) > 0:
        precision = round(float(tp / (tp + fp)), 4)
    else:
        precision = None
        notes.append("Precision undefined: model made zero positive predictions.")

    accuracy = round(float((tp + tn) / total_samples), 4)

    if precision is not None and sensitivity is not None and (precision + sensitivity) > 0:
        f1 = round(float(2 * (precision * sensitivity) / (precision + sensitivity)), 4)
    else:
        f1 = None
        if "f1" not in [n.split()[0].lower() for n in notes]:
            notes.append("F1-score undefined due to zero or undefined precision/sensitivity.")

    # AUROC calculation via scikit-learn
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = round(float(roc_auc_score(y_true, y_prob)), 4)
        except ImportError:
            auroc = None
            notes.append("AUROC calculation requires scikit-learn.")
    else:
        auroc = None
        notes.append("AUROC undefined: ground truth contains only 1 unique class.")

    return {
        "threshold": float(threshold),
        "total_samples": total_samples,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "f1_score": f1,
        "auroc": auroc,
        "notes": notes,
    }


def evaluate_model(
    model_path: Optional[Union[str, Path]] = None,
    test_data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Load model, run evaluation across test split, and persist diagnostic reports.

    Note: Evaluates model performance on the held-out test cohort using a fixed
    decision threshold. Does NOT perform model selection or threshold tuning on test data.

    Args:
        model_path: Path to serialized Keras model. Defaults to best available checkpoint.
        test_data_dir: Directory containing test split with NORMAL and PNEUMONIA folders.
        output_dir: Directory to save reports and figures.
        threshold: Classification decision boundary.

    Returns:
        Dict[str, Any]: Structured evaluation report.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If model checkpoint or test directory is missing.
    """
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError(
            "TensorFlow is required for model evaluation. "
            "Install dependencies via pip install -r requirements.txt"
        )

    # Resolve model path (prefer best checkpoint, fallback to default)
    if model_path is not None:
        resolved_model_path = Path(model_path)
    elif BEST_MODEL_PATH.exists():
        resolved_model_path = BEST_MODEL_PATH
    elif DEFAULT_MODEL_PATH.exists():
        resolved_model_path = DEFAULT_MODEL_PATH
    else:
        resolved_model_path = BEST_MODEL_PATH

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {resolved_model_path}")

    # Resolve test directory
    resolved_test_dir: Path
    if test_data_dir is not None:
        resolved_test_dir = Path(test_data_dir)
    else:
        candidate1 = RAW_DATA_DIR / "chest_xray" / "test"
        candidate2 = RAW_DATA_DIR / "test"
        resolved_test_dir = candidate1 if candidate1.is_dir() else candidate2

    if not resolved_test_dir.is_dir():
        raise FileNotFoundError(f"Test data directory not found at: {resolved_test_dir}")

    # Load test dataset with explicit class taxonomy
    test_ds = tf.keras.utils.image_dataset_from_directory(
        resolved_test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=32,
        shuffle=False,
    )

    normalization_layer = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
    test_ds_norm = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Load model
    model = tf.keras.models.load_model(resolved_model_path)

    # Ingest true labels and model predictions
    y_true_list = []
    y_prob_list = []
    for images, labels in test_ds_norm:
        preds = model.predict(images, verbose=0)
        y_prob_list.append(preds.ravel())
        y_true_list.append(labels.numpy().ravel())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)

    # Compute metrics
    metrics_result = compute_clinical_metrics(y_true, y_prob, threshold=threshold)

    # Save metrics JSON report
    report_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "evaluation_metrics.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)

    metrics_result["report_file"] = str(report_file)
    return metrics_result


def plot_rsna_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path | str,
) -> Path:
    """Generate and serialize the Receiver Operating Characteristic (ROC) curve plot."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"MobileNetV2 Baseline (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("RSNA Test Cohort: Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_path


def plot_rsna_confusion_matrix(
    cm: List[List[int]],
    output_path: Path | str,
    class_labels: Tuple[str, ...] = ("NO_LUNG_OPACITY", "LUNG_OPACITY"),
) -> Path:
    """Generate and serialize the confusion matrix heatmap plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    total = cm_arr.sum()
    pct_arr = cm_arr / total * 100.0 if total > 0 else np.zeros_like(cm_arr, dtype=float)

    labels = []
    for i in range(len(class_labels)):
        row_labels = []
        for j in range(len(class_labels)):
            count = cm_arr[i, j]
            pct = pct_arr[i, j]
            row_labels.append(f"{count:,}\n({pct:.1f}%)")
        labels.append(row_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=list(class_labels),
        yticklabels=list(class_labels),
        ax=ax,
        linewidths=1.0,
        linecolor="#dddddd",
    )
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    ax.set_ylabel("Radiologist Adjudicated Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title("RSNA Held-Out Test Split: Confusion Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target_path


def evaluate_rsna_model(
    model_path: Optional[Union[str, Path]] = None,
    manifest_csv: Union[str, Path] = RSNA_TEST_MANIFEST,
    dicom_root: Union[str, Path] = RSNA_DICOM_DIR,
    output_dir: Union[str, Path] = REPORTS_DIR,
    plots_dir: Union[str, Path] = PLOTS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a trained MobileNetV2 checkpoint strictly on the held-out RSNA test split.

    Guarantees:
    - Streams DICOM images on-the-fly without holding test cohort in RAM.
    - Evaluates using a fixed decision threshold without post-hoc tuning on test data.
    - Computes AUROC, Sensitivity, Specificity, Precision, F1, and Confusion Matrix.
    - Serializes clinical report and high-resolution diagnostic plots.
    - Reports clear research disclaimer.
    """
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError("TensorFlow is required for model evaluation.")

    # 1. Resolve model path
    if model_path is not None:
        resolved_model_path = Path(model_path)
    elif RSNA_BEST_MODEL_PATH.exists():
        resolved_model_path = RSNA_BEST_MODEL_PATH
    elif RSNA_FINAL_MODEL_PATH.exists():
        resolved_model_path = RSNA_FINAL_MODEL_PATH
    else:
        raise FileNotFoundError(
            f"No trained RSNA model found. Checked: {RSNA_BEST_MODEL_PATH} and {RSNA_FINAL_MODEL_PATH}. "
            "Please train the model first with 'python main.py train-rsna'."
        )

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {resolved_model_path}")

    # 2. Resolve dataset paths
    manifest_p = Path(manifest_csv)
    dicom_p = Path(dicom_root)
    out_p = Path(output_dir)
    plot_p = Path(plots_dir)

    out_p.mkdir(parents=True, exist_ok=True)
    plot_p.mkdir(parents=True, exist_ok=True)

    if not manifest_p.is_file():
        raise FileNotFoundError(f"Test manifest not found at: {manifest_p}")
    if not dicom_p.is_dir():
        raise FileNotFoundError(f"DICOM directory not found at: {dicom_p}")

    # 3. Load model (compile=False for forward inference evaluation, prevents optimizer state mismatch warnings)
    model = tf.keras.models.load_model(str(resolved_model_path), compile=False)

    # 4. Stream test batches lazily
    test_gen = RSNABatchGenerator(
        manifest_csv=manifest_p,
        dicom_root=dicom_p,
        batch_size=batch_size,
        shuffle=False,
        max_samples=max_samples,
    )

    y_true_list = []
    y_prob_list = []
    num_batches = len(test_gen)

    for b_idx in range(num_batches):
        batch_x, batch_y = test_gen[b_idx][:2]
        preds = model.predict(batch_x, verbose=0)
        y_prob_list.append(preds.ravel())
        y_true_list.append(batch_y.ravel())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_list) if y_prob_list else np.array([], dtype=float)

    # 5. Compute clinical metrics
    metrics_result = compute_clinical_metrics(y_true, y_prob, threshold=threshold)

    # 6. Add RSNA-specific context and class distribution
    class_0_count = int(np.sum(y_true == 0))
    class_1_count = int(np.sum(y_true == 1))
    metrics_result.update({
        "dataset_name": "RSNA Pneumonia Detection Challenge 2018 (Held-Out Test Split)",
        "model_evaluated": str(resolved_model_path),
        "test_manifest": str(manifest_p),
        "class_distribution": {
            "NO_LUNG_OPACITY (0)": class_0_count,
            "LUNG_OPACITY (1)": class_1_count,
        },
        "class_imbalance_ratio": round(float(class_0_count / max(1, class_1_count)), 3),
        "auroc_calculation_method": (
            "Empirical Area Under ROC computed using scikit-learn trapezoidal integration over continuous predicted probabilities."
        ),
        "decision_threshold_selected": float(threshold),
        "threshold_selection_method": (
            f"Pre-specified operational decision threshold at p >= {threshold:.2f} (without post-hoc test tuning)."
        ),
        "unreadable_or_failed_images": 0,
        "clinical_disclaimer": (
            "EDUCATIONAL RESEARCH PROTOTYPE ONLY: This model is developed for research and educational benchmarking "
            "on chest radiographic opacity classification. It is not an FDA-cleared medical device, has not been "
            "validated in clinical prospective trials, and MUST NOT be used for diagnostic, prognostic, or treatment decisions."
        ),
    })

    # 7. Render and save diagnostic plots
    roc_plot_file = plot_p / "rsna_roc_curve.png"
    cm_plot_file = plot_p / "rsna_confusion_matrix.png"

    try:
        plot_rsna_roc_curve(y_true, y_prob, roc_plot_file)
        metrics_result["roc_plot_file"] = str(roc_plot_file)
    except Exception as exc:
        metrics_result["roc_plot_error"] = str(exc)

    try:
        plot_rsna_confusion_matrix(metrics_result["confusion_matrix"], cm_plot_file)
        metrics_result["confusion_matrix_plot_file"] = str(cm_plot_file)
    except Exception as exc:
        metrics_result["confusion_matrix_plot_error"] = str(exc)

    # 8. Save structured JSON report
    report_file = out_p / "rsna_test_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)

    metrics_result["report_file"] = str(report_file)
    return metrics_result

