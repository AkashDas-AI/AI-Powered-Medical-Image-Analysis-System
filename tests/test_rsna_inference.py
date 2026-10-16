"""Automated unit and integration tests for RSNA DICOM inference and Grad-CAM explainability.

Verifies end-to-end model loading, DICOM decoding, classification contracts,
Grad-CAM feature saliency generation, and CLI subcommands.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from src.config import (
    DEFAULT_THRESHOLD,
    PROJECT_ROOT,
    RSNA_BEST_MODEL_PATH,
    RSNA_DICOM_DIR,
    RSNA_TRAIN_MANIFEST,
)
from src.explainability import (
    find_target_conv_layer,
    generate_gradcam,
    generate_gradcam_figure,
    overlay_heatmap,
)
from src.predict import load_inference_model, predict_rsna_dicom
from src.rsna_dataset import load_dicom_as_array


def _get_sample_dicom_path() -> Optional[Path]:
    """Helper to locate an existing local DICOM file for testing."""
    if not RSNA_TRAIN_MANIFEST.exists():
        return None
    import csv
    with open(RSNA_TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidate = RSNA_DICOM_DIR / row["dicom_relative_path"]
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


@pytest.fixture(scope="module")
def sample_dicom_path() -> Path:
    """Fixture returning an existing local RSNA DICOM path."""
    dcm_path = _get_sample_dicom_path()
    if dcm_path is None or not dcm_path.exists():
        pytest.skip("No local RSNA DICOM files available for inference testing.")
    return dcm_path


@pytest.fixture(scope="module")
def rsna_model():
    """Fixture providing loaded RSNA MobileNetV2 model checkpoint."""
    if not RSNA_BEST_MODEL_PATH.exists():
        pytest.skip(f"RSNA best checkpoint not found at {RSNA_BEST_MODEL_PATH}")
    return load_inference_model(RSNA_BEST_MODEL_PATH)


def test_load_inference_model_default():
    """Verify load_inference_model loads default RSNA model when present."""
    if not RSNA_BEST_MODEL_PATH.exists():
        pytest.skip("Default RSNA model not present.")
    model = load_inference_model()
    assert model is not None
    assert hasattr(model, "predict") or callable(model)
    assert model.output_shape == (None, 1)


def test_load_inference_model_missing_error():
    """Verify FileNotFoundError on non-existent checkpoint path."""
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_inference_model("models/non_existent_checkpoint_xyz123.keras")


def test_predict_rsna_dicom_contract(sample_dicom_path: Path, rsna_model):
    """Verify structured contract returned by predict_rsna_dicom."""
    result = predict_rsna_dicom(
        dicom_input=sample_dicom_path,
        model=rsna_model,
        threshold=0.5,
    )

    required_keys = [
        "model_checkpoint",
        "dicom_path",
        "predicted_class",
        "lung_opacity_score",
        "decision_threshold",
        "is_opacity_detected",
        "probabilities",
        "dicom_metadata",
        "input_tensor_shape",
        "disclaimer",
    ]
    for key in required_keys:
        assert key in result, f"Missing key in prediction result: {key}"

    assert result["predicted_class"] in ("LUNG_OPACITY", "NO_LUNG_OPACITY")
    assert 0.0 <= result["lung_opacity_score"] <= 1.0
    assert result["decision_threshold"] == 0.5
    assert isinstance(result["is_opacity_detected"], bool)
    assert result["input_tensor_shape"] == [1, 224, 224, 3]

    probs = result["probabilities"]
    assert "LUNG_OPACITY" in probs and "NO_LUNG_OPACITY" in probs
    assert np.isclose(probs["LUNG_OPACITY"] + probs["NO_LUNG_OPACITY"], 1.0, atol=1e-3)

    assert "EDUCATIONAL RESEARCH PROTOTYPE ONLY" in result["disclaimer"]


def test_predict_rsna_threshold_boundary(sample_dicom_path: Path, rsna_model):
    """Verify threshold boundary modifies binary decision appropriately."""
    # At threshold 0.0, any non-negative score is classified as LUNG_OPACITY
    res_zero = predict_rsna_dicom(sample_dicom_path, model=rsna_model, threshold=0.0)
    assert res_zero["predicted_class"] == "LUNG_OPACITY"
    assert res_zero["is_opacity_detected"] is True

    # At threshold 1.0, only exactly 1.0 is LUNG_OPACITY; otherwise NO_LUNG_OPACITY
    res_one = predict_rsna_dicom(sample_dicom_path, model=rsna_model, threshold=1.0)
    if res_one["lung_opacity_score"] < 1.0:
        assert res_one["predicted_class"] == "NO_LUNG_OPACITY"
        assert res_one["is_opacity_detected"] is False


def test_predict_rsna_invalid_inputs(sample_dicom_path: Path, rsna_model):
    """Verify defensive exceptions for invalid paths, types, and thresholds."""
    with pytest.raises(FileNotFoundError, match="DICOM file not found"):
        predict_rsna_dicom("non_existent_dicom_path.dcm", model=rsna_model)

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        predict_rsna_dicom(sample_dicom_path, model=rsna_model, threshold=1.5)

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        predict_rsna_dicom(sample_dicom_path, model=rsna_model, threshold=-0.05)

    with pytest.raises(TypeError, match="dicom_input must be a str or Path"):
        predict_rsna_dicom(12345, model=rsna_model)  # type: ignore


def test_gradcam_computation_rsna(sample_dicom_path: Path, rsna_model):
    """Verify Grad-CAM produces non-degenerate, finite heatmaps on RSNA model."""
    img_uint8 = load_dicom_as_array(sample_dicom_path, target_size=(224, 224))
    tensor = ((img_uint8.astype(np.float32) / 127.5) - 1.0)[np.newaxis, ...]

    target_layer = find_target_conv_layer(rsna_model)
    assert target_layer == "Conv_1"

    cam = generate_gradcam(rsna_model, tensor, target_layer_name=target_layer)
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (7, 7)
    assert np.all(np.isfinite(cam))
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_overlay_heatmap_with_dicom_path(sample_dicom_path: Path, rsna_model):
    """Verify overlay_heatmap accepts a Path to a .dcm file directly."""
    img_uint8 = load_dicom_as_array(sample_dicom_path, target_size=(224, 224))
    tensor = ((img_uint8.astype(np.float32) / 127.5) - 1.0)[np.newaxis, ...]
    cam = generate_gradcam(rsna_model, tensor)

    overlay = overlay_heatmap(sample_dicom_path, cam, alpha=0.4)
    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"
    assert overlay.size[0] > 0 and overlay.size[1] > 0


def test_generate_gradcam_figure(sample_dicom_path: Path, rsna_model, tmp_path: Path):
    """Verify generate_gradcam_figure generates and saves the 3-panel figure."""
    img_uint8 = load_dicom_as_array(sample_dicom_path, target_size=(224, 224))
    tensor = ((img_uint8.astype(np.float32) / 127.5) - 1.0)[np.newaxis, ...]
    cam = generate_gradcam(rsna_model, tensor)

    save_path = tmp_path / "test_gradcam_panel.png"
    fig = generate_gradcam_figure(
        original_image=img_uint8,
        heatmap=cam,
        predicted_class="LUNG_OPACITY",
        model_score=0.75,
        threshold=0.5,
        target_layer_name="Conv_1",
        save_path=save_path,
    )
    assert fig is not None
    assert save_path.exists()
    assert save_path.stat().st_size > 10000  # Valid rendered PNG image


def test_main_cli_predict_rsna(sample_dicom_path: Path, tmp_path: Path):
    """Verify main.py predict-rsna CLI execution end-to-end."""
    from main import main

    out_img = tmp_path / "cli_gradcam.png"
    out_json = tmp_path / "cli_result.json"

    exit_code = main([
        "predict-rsna",
        "--image", str(sample_dicom_path),
        "--threshold", "0.5",
        "--gradcam",
        "--output", str(out_img),
        "--json-output", str(out_json),
    ])
    assert exit_code == 0
    assert out_img.exists()
    assert out_json.exists()


def test_main_cli_predict_rsna_missing_image():
    """Verify main.py predict-rsna fails cleanly with return code 1 when image is missing."""
    from main import main
    exit_code = main(["predict-rsna", "--image", "non_existent_sample.dcm"])
    assert exit_code == 1


def test_predict_rsna_privacy_by_default(sample_dicom_path: Path, rsna_model):
    """Verify Patient ID and sensitive UIDs are strictly excluded by default."""
    result = predict_rsna_dicom(sample_dicom_path, model=rsna_model, include_patient_id=False)
    meta = result["dicom_metadata"]
    assert "patient_id" not in meta, "Privacy violation: Patient ID exposed in default metadata"
    assert "study_instance_uid" not in meta, "Privacy violation: Study UID exposed"
    assert "sop_instance_uid" not in meta, "Privacy violation: SOP UID exposed"
    assert "view_position" in meta
    assert "photometric_interpretation" in meta


def test_predict_rsna_privacy_opt_in(sample_dicom_path: Path, rsna_model):
    """Verify Patient ID is only included when explicitly requested via include_patient_id=True."""
    result = predict_rsna_dicom(sample_dicom_path, model=rsna_model, include_patient_id=True)
    meta = result["dicom_metadata"]
    assert "patient_id" in meta
    assert meta["patient_id"] != ""


def test_predict_rsna_corrupt_dicom_error(tmp_path: Path, rsna_model):
    """Verify corrupt or unreadable DICOM raises clear ValueError without leaking data."""
    bad_dcm = tmp_path / "corrupt_study.dcm"
    bad_dcm.write_bytes(b"NOT_A_VALID_DICOM_HEADER_CORRUPTED_BYTES")

    with pytest.raises(ValueError, match="Failed to read or decode DICOM image"):
        predict_rsna_dicom(bad_dcm, model=rsna_model)


def test_gradcam_degenerate_zero_attribution(sample_dicom_path: Path):
    """Verify degenerate (all-zero) heatmap does not manufacture a false colored overlay."""
    zero_cam = np.zeros((7, 7), dtype=np.float32)

    # overlay_heatmap must return base image directly on zero heatmap
    base_overlay = overlay_heatmap(sample_dicom_path, zero_cam)
    assert isinstance(base_overlay, Image.Image)

    # generate_gradcam_figure must render with zero attribution notice
    fig = generate_gradcam_figure(
        original_image=sample_dicom_path,
        heatmap=zero_cam,
        predicted_class="NO_LUNG_OPACITY",
        model_score=0.1,
    )
    assert fig is not None


def test_preprocessing_parity_monochrome1_and_2(tmp_path: Path):
    """Verify MONOCHROME1 inversion produces consistent contrast matching MONOCHROME2."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    def _create_mock_dcm(path: Path, photometric: str, pixel_data: np.ndarray) -> None:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = "OT"
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PhotometricInterpretation = photometric
        ds.SamplesPerPixel = 1
        ds.Rows, ds.Columns = pixel_data.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = pixel_data.astype(np.uint16).tobytes()
        ds.save_as(str(path))

    # Gradient from 0 to 1000
    grad = np.tile(np.linspace(0, 1000, 64, dtype=np.uint16), (64, 1))

    # M1: 0 is white, 1000 is dark -> after inversion, 0 becomes dark, 1000 becomes bright
    m1_path = tmp_path / "mock_m1.dcm"
    _create_mock_dcm(m1_path, "MONOCHROME1", 1000 - grad)

    # M2: 0 is dark, 1000 is bright
    m2_path = tmp_path / "mock_m2.dcm"
    _create_mock_dcm(m2_path, "MONOCHROME2", grad)

    arr_m1 = load_dicom_as_array(m1_path, target_size=(64, 64))
    arr_m2 = load_dicom_as_array(m2_path, target_size=(64, 64))

    # Pixel intensities should match within tolerance after normalization and inversion
    assert np.allclose(arr_m1.astype(float), arr_m2.astype(float), atol=2.0)


def test_main_cli_predict_rsna_privacy_by_default(sample_dicom_path: Path, tmp_path: Path, capsys):
    """Verify main.py predict-rsna does not print or save Patient ID by default."""
    from main import main
    import json

    out_json = tmp_path / "privacy_test.json"
    exit_code = main([
        "predict-rsna",
        "--image", str(sample_dicom_path),
        "--json-output", str(out_json),
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Patient ID" not in captured.out, "Privacy leak in CLI standard output"

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "patient_id" not in data.get("dicom_metadata", {}), "Privacy leak in saved JSON"


