"""Unit and integration tests for RSNA MobileNetV2 training and evaluation pipeline.

Verifies:
1. RSNABatchGenerator initialization, batch shapes, normalization, and sample weights.
2. Balanced class weight calculation derived strictly from training split distributions.
3. Smoke-test workflow execution (forward pass, finite loss, training step, checkpointing).
4. RSNA test evaluation reporting, clinical metrics, and diagnostic plot serialization.
"""

import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.config import (
    RANDOM_SEED,
    RSNA_DICOM_DIR,
    RSNA_TEST_MANIFEST,
    RSNA_TRAIN_MANIFEST,
    RSNA_VAL_MANIFEST,
)
from src.evaluate import evaluate_rsna_model, plot_rsna_confusion_matrix, plot_rsna_roc_curve
from src.model import build_mobilenet_v2
from src.rsna_dataset import RSNABatchGenerator
from src.train import plot_training_curves, run_rsna_smoke_test, train_rsna_baseline


def _create_synthetic_dicom(
    file_path: Path,
    pixel_array: np.ndarray,
    patient_id: str = "PID_TEST_001",
) -> Path:
    """Helper creating a minimal valid uncompressed DICOM on disk for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(file_path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = pixel_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(str(file_path))
    return file_path


@pytest.fixture
def synthetic_rsna_environment(tmp_path: Path):
    """Fixture providing synthetic DICOM files and manifest CSVs for testing."""
    if not PYDICOM_AVAILABLE:
        pytest.skip("pydicom is required for RSNA test suite.")

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir(parents=True)

    # Create 8 synthetic DICOM images: 6 normal (0), 2 opacity (1)
    train_records = []
    for i in range(8):
        grid = np.full((64, 64), fill_value=20 * (i + 1), dtype=np.uint8)
        sop = f"train_sop_{i:03d}"
        dcm_p = _create_synthetic_dicom(dicom_dir / f"{sop}.dcm", grid, patient_id=f"PAT_{i}")
        label = 1 if i in (2, 5) else 0
        train_records.append({
            "sop_instance_uid": sop,
            "patient_id": f"PAT_{i}",
            "label_name": "LUNG_OPACITY" if label == 1 else "NO_LUNG_OPACITY",
            "label_index": label,
            "dicom_relative_path": f"{sop}.dcm",
        })

    train_manifest = tmp_path / "train_manifest.csv"
    with open(train_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_records[0].keys()))
        writer.writeheader()
        writer.writerows(train_records)

    # Create 4 synthetic validation images
    val_records = []
    for i in range(4):
        grid = np.full((64, 64), fill_value=40 * (i + 1), dtype=np.uint8)
        sop = f"val_sop_{i:03d}"
        _create_synthetic_dicom(dicom_dir / f"{sop}.dcm", grid, patient_id=f"VAL_PAT_{i}")
        label = 1 if i == 1 else 0
        val_records.append({
            "sop_instance_uid": sop,
            "patient_id": f"VAL_PAT_{i}",
            "label_name": "LUNG_OPACITY" if label == 1 else "NO_LUNG_OPACITY",
            "label_index": label,
            "dicom_relative_path": f"{sop}.dcm",
        })

    val_manifest = tmp_path / "val_manifest.csv"
    with open(val_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_records[0].keys()))
        writer.writeheader()
        writer.writerows(val_records)

    return {
        "dicom_dir": dicom_dir,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_count": 8,
        "val_count": 4,
    }


def test_rsna_batch_generator_properties_and_class_weights(synthetic_rsna_environment):
    """Verify RSNABatchGenerator properties, sample count, and balanced class weight formula."""
    env = synthetic_rsna_environment
    gen = RSNABatchGenerator(
        manifest_csv=env["train_manifest"],
        dicom_root=env["dicom_dir"],
        batch_size=4,
        shuffle=False,
    )

    assert gen.total_samples == 8
    assert len(gen) == 2  # 8 samples / batch size 4 = 2 batches
    assert gen.class_counts == {0: 6, 1: 2}

    # Verify class weights formula: w_j = N / (K * N_j)
    # w_0 = 8 / (2 * 6) = 8 / 12 = 0.6667
    # w_1 = 8 / (2 * 2) = 8 / 4  = 2.0000
    weights = gen.compute_balanced_class_weights()
    assert pytest.approx(weights[0], rel=1e-3) == (8.0 / 12.0)
    assert pytest.approx(weights[1], rel=1e-3) == 2.0


def test_rsna_batch_generator_indexing_and_shapes(synthetic_rsna_environment):
    """Verify batch indexing returns valid tensor shapes and MobileNetV2 pixel bounds."""
    env = synthetic_rsna_environment
    gen = RSNABatchGenerator(
        manifest_csv=env["train_manifest"],
        dicom_root=env["dicom_dir"],
        batch_size=4,
        target_size=(224, 224),
        shuffle=False,
    )

    batch_x, batch_y = gen[0]
    assert batch_x.shape == (4, 224, 224, 3)
    assert batch_x.dtype == np.float32
    assert -1.0 <= batch_x.min() and batch_x.max() <= 1.0
    assert batch_y.shape == (4,)
    assert batch_y.dtype == np.int32
    assert np.isfinite(batch_x).all()


def test_rsna_batch_generator_sample_weights(synthetic_rsna_environment):
    """Verify generator yields sample weights when class_weights dictionary is supplied."""
    env = synthetic_rsna_environment
    c_weights = {0: 0.5, 1: 2.5}
    gen = RSNABatchGenerator(
        manifest_csv=env["train_manifest"],
        dicom_root=env["dicom_dir"],
        batch_size=4,
        class_weights=c_weights,
        shuffle=False,
    )

    batch_x, batch_y, batch_sw = gen[0]
    assert len(batch_sw) == 4
    for y_val, sw_val in zip(batch_y, batch_sw):
        assert pytest.approx(sw_val) == c_weights[int(y_val)]


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow is required for smoke test")
def test_rsna_smoke_test_execution(synthetic_rsna_environment, tmp_path: Path):
    """Verify run_rsna_smoke_test completes forward pass, backward step, and saves checkpoint."""
    env = synthetic_rsna_environment
    smoke_ckpt = tmp_path / "smoke_model.keras"

    result = run_rsna_smoke_test(
        train_manifest=env["train_manifest"],
        dicom_root=env["dicom_dir"],
        weights=None,  # Offline initialization
        batch_size=4,
        checkpoint_path=smoke_ckpt,
    )

    assert result["status"] == "PASS"
    assert result["input_shape"] == [None, 224, 224, 3]
    assert result["output_shape"] == [None, 1]
    assert result["batch_size_tested"] == 4
    assert np.isfinite(result["loss_after_one_step"])
    assert result["loss_after_one_step"] > 0
    assert smoke_ckpt.is_file()


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow is required for training test")
def test_train_rsna_baseline_minimal_run(synthetic_rsna_environment, tmp_path: Path):
    """Verify train_rsna_baseline executes 1 epoch, saves model, config, history, and plot."""
    env = synthetic_rsna_environment
    final_model = tmp_path / "final_rsna.keras"
    best_model = tmp_path / "best_rsna.keras"
    out_dir = tmp_path / "reports"
    plot_dir = tmp_path / "plots"

    result = train_rsna_baseline(
        train_manifest=env["train_manifest"],
        val_manifest=env["val_manifest"],
        dicom_root=env["dicom_dir"],
        model_save_path=final_model,
        best_model_path=best_model,
        epochs=1,
        batch_size=4,
        freeze_base=True,
        weights=None,  # Offline initialization
        steps_per_epoch=1,
        val_steps=1,
        output_dir=out_dir,
        plots_dir=plot_dir,
    )

    assert result["status"] == "success"
    assert result["epochs_completed"] == 1
    assert result["experiment_type"] == "STEP_LIMITED_PILOT"
    assert final_model.is_file()
    assert (out_dir / "rsna_training_config.json").is_file()
    assert (out_dir / "rsna_training_history.csv").is_file()
    assert (plot_dir / "rsna_training_history.png").is_file()


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow is required for training test")
def test_train_rsna_baseline_full_pass(synthetic_rsna_environment, tmp_path: Path):
    """Verify train_rsna_baseline correctly tags full pass coverage when steps are unlimited."""
    env = synthetic_rsna_environment
    final_model = tmp_path / "final_rsna_full.keras"
    best_model = tmp_path / "best_rsna_full.keras"
    out_dir = tmp_path / "reports_full"
    plot_dir = tmp_path / "plots_full"

    result = train_rsna_baseline(
        train_manifest=env["train_manifest"],
        val_manifest=env["val_manifest"],
        dicom_root=env["dicom_dir"],
        model_save_path=final_model,
        best_model_path=best_model,
        epochs=1,
        batch_size=4,
        freeze_base=True,
        weights=None,
        steps_per_epoch=None,
        val_steps=None,
        output_dir=out_dir,
        plots_dir=plot_dir,
    )

    assert result["status"] == "success"
    assert result["experiment_type"] == "FULL_PASS_EXPERIMENT"
    assert result["is_full_training_pass"] is True
    assert result["is_full_validation_pass"] is True
    assert result["train_coverage_percentage"] == 100.0
    assert result["val_coverage_percentage"] == 100.0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow is required for evaluation test")
def test_evaluate_rsna_model_pipeline(synthetic_rsna_environment, tmp_path: Path):
    """Verify evaluate_rsna_model generates clinical metrics, JSON report, and ROC/CM plots."""
    env = synthetic_rsna_environment

    # Save a dummy unweighted model
    dummy_model = build_mobilenet_v2(weights=None)
    model_path = tmp_path / "eval_model.keras"
    dummy_model.save(str(model_path))

    out_dir = tmp_path / "eval_reports"
    plot_dir = tmp_path / "eval_plots"

    report = evaluate_rsna_model(
        model_path=model_path,
        manifest_csv=env["val_manifest"],
        dicom_root=env["dicom_dir"],
        output_dir=out_dir,
        plots_dir=plot_dir,
        threshold=0.5,
        batch_size=2,
    )

    assert report["total_samples"] == 4
    assert "sensitivity_recall" in report
    assert "specificity" in report
    assert "auroc" in report
    assert (out_dir / "rsna_test_evaluation.json").is_file()
    assert (plot_dir / "rsna_roc_curve.png").is_file()
    assert (plot_dir / "rsna_confusion_matrix.png").is_file()
    assert "EDUCATIONAL RESEARCH PROTOTYPE" in report["clinical_disclaimer"]

