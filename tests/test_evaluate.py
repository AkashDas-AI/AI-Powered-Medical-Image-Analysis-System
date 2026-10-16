"""Unit tests for the evaluation metrics module and clinical calculation edge cases.

Verifies calculation of Sensitivity, Specificity, AUROC, F1-Score, Confusion Matrix,
and safe handling of undefined-metric edge cases (single-class cohorts, zero positives).
"""

import numpy as np
import pytest

from src.evaluate import compute_clinical_metrics


def test_compute_metrics_standard_case() -> None:
    """Verify metrics calculation on a standard cohort with both positive and negative cases."""
    # 4 samples: 2 normal (0), 2 pneumonia (1)
    y_true = np.array([0, 0, 1, 1])
    # Predicted probabilities: correctly predicts all 4 with threshold 0.5
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_clinical_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["total_samples"] == 4
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["true_negatives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["sensitivity_recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_compute_metrics_imperfect_predictions() -> None:
    """Verify metrics with mixed false positives and false negatives."""
    # 4 samples: 2 normal (0), 2 pneumonia (1)
    y_true = np.array([0, 0, 1, 1])
    # Model predicts: sample 0 -> TP? no, label is 0 so pred=1 is FP
    # sample 1 -> pred=0 (TN)
    # sample 2 -> pred=1 (TP)
    # sample 3 -> pred=0 (FN)
    y_prob = np.array([0.8, 0.2, 0.9, 0.3])

    metrics = compute_clinical_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["sensitivity_recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["f1_score"] == 0.5


def test_compute_metrics_undefined_sensitivity_zero_positives() -> None:
    """Verify handling when cohort contains zero positive cases (all normal)."""
    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])

    metrics = compute_clinical_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["sensitivity_recall"] is None
    assert metrics["specificity"] == 1.0
    assert metrics["auroc"] is None  # Single class ground truth
    assert any("Sensitivity undefined" in n for n in metrics["notes"])
    assert any("AUROC undefined" in n for n in metrics["notes"])


def test_compute_metrics_undefined_specificity_zero_negatives() -> None:
    """Verify handling when cohort contains zero negative cases (all pneumonia)."""
    y_true = np.array([1, 1, 1])
    y_prob = np.array([0.8, 0.9, 0.7])

    metrics = compute_clinical_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["sensitivity_recall"] == 1.0
    assert metrics["specificity"] is None
    assert metrics["auroc"] is None
    assert any("Specificity undefined" in n for n in metrics["notes"])


def test_compute_metrics_empty_array() -> None:
    """Verify empty input arrays return zero counts and None metrics without crashing."""
    metrics = compute_clinical_metrics(np.array([]), np.array([]))

    assert metrics["total_samples"] == 0
    assert metrics["sensitivity_recall"] is None
    assert metrics["accuracy"] is None
    assert metrics["confusion_matrix"] == [[0, 0], [0, 0]]

