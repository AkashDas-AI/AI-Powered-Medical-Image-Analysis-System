"""Unit and Integration Tests for RSNA Pneumonia Detection Challenge Pipeline.

Covers:
- NIH patient ID parsing from image filenames.
- Mapping loading, field validation, and duplicate detection.
- Annotation-to-label mapping and bounding-box parsing.
- Exclusion of unsupported, ambiguous, and radiologist-flagged records.
- Patient-level split disjointness (zero patient leakage).
- Deterministic manifest generation with a fixed random seed.
- Missing DICOM file handling and malformed metadata.
- Preprocessing and MONOCHROME1 inversion.
- Memory-safe streaming batch generator contract.
- End-to-end RSNA audit pipeline execution.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
except ImportError:
    pydicom = None

from src.rsna_dataset import (
    CALCULATED_LABEL_NORMAL_ID,
    CALCULATED_LABEL_NOT_NORMAL_ID,
    CALCULATED_LABEL_OPACITY_ID,
    LABEL_LUNG_OPACITY,
    LABEL_NO_LUNG_OPACITY,
    RSNA_LABEL_TO_INT,
    RSNARecord,
    audit_rsna_pipeline,
    build_cohort_records,
    build_dicom_index,
    create_patient_level_splits,
    extract_patient_id,
    load_dicom_as_array,
    load_rsna_annotations,
    load_rsna_mapping,
    resolve_dicom_path,
    rsna_batch_generator,
    save_manifests,
    validate_local_dicoms,
)


def create_minimal_dicom_file(
    file_path: Path,
    pixel_array: np.ndarray,
    photometric: str = "MONOCHROME2",
    patient_id: str = "TEST_PATIENT"
) -> Path:
    """Helper to generate a minimal valid DICOM file on disk for testing."""
    if pydicom is None:
        pytest.skip("pydicom is not installed.")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "CR"
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = pixel_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PixelData = pixel_array.astype(np.uint8).tobytes()

    ds.save_as(str(file_path))
    return file_path


# ==============================================================================
# 1. Patient ID Parsing Tests
# ==============================================================================

def test_extract_patient_id_valid_patterns():
    """Verify standard 8-digit NIH CXR8 filenames resolve patient IDs."""
    assert extract_patient_id("00012773_000.png") == "00012773"
    assert extract_patient_id("00000001_005.png") == "00000001"
    assert extract_patient_id("12345678_012.png") == "12345678"


def test_extract_patient_id_invalid_patterns():
    """Verify malformed or empty image filenames safely return None."""
    assert extract_patient_id("") is None
    assert extract_patient_id(None) is None  # type: ignore
    assert extract_patient_id("invalid_filename.png") is None
    assert extract_patient_id("abc_000.png") is None


# ==============================================================================
# 2. Mapping Loading & Validation Tests
# ==============================================================================

def test_load_rsna_mapping_valid(tmp_path: Path):
    """Verify loading valid mapping entries and patient ID extraction."""
    mapping_data = [
        {
            "img_id": "00010001_000.png",
            "subset_img_id": "uuid-1",
            "StudyInstanceUID": "study-1",
            "SeriesInstanceUID": "series-1",
            "SOPInstanceUID": "sop-1"
        },
        {
            "img_id": "00010002_001.png",
            "subset_img_id": "uuid-2",
            "StudyInstanceUID": "study-2",
            "SeriesInstanceUID": "series-2",
            "SOPInstanceUID": "sop-2"
        }
    ]
    mapping_file = tmp_path / "mappings.json"
    mapping_file.write_text(json.dumps(mapping_data), encoding="utf-8")

    records, errors = load_rsna_mapping(mapping_file)
    assert len(records) == 2
    assert len(errors) == 0
    assert records[0]["patient_id"] == "00010001"
    assert records[1]["patient_id"] == "00010002"


def test_load_rsna_mapping_duplicate_and_missing_fields(tmp_path: Path):
    """Verify detection of duplicate SOPInstanceUID and missing required fields."""
    mapping_data = [
        {
            "img_id": "00010001_000.png",
            "StudyInstanceUID": "study-1",
            "SOPInstanceUID": "sop-1"
        },
        {  # Duplicate SOP
            "img_id": "00010001_001.png",
            "StudyInstanceUID": "study-1b",
            "SOPInstanceUID": "sop-1"
        },
        {  # Missing SOPInstanceUID
            "img_id": "00010003_000.png",
            "StudyInstanceUID": "study-3"
        },
        {  # Malformed img_id
            "img_id": "non_standard.png",
            "StudyInstanceUID": "study-4",
            "SOPInstanceUID": "sop-4"
        }
    ]
    mapping_file = tmp_path / "mappings_err.json"
    mapping_file.write_text(json.dumps(mapping_data), encoding="utf-8")

    records, errors = load_rsna_mapping(mapping_file)
    assert len(records) == 1
    assert "sop-1" in errors
    assert "sop-4" in errors


# ==============================================================================
# 3. Annotation Schema & Label Mapping Tests
# ==============================================================================

def test_load_rsna_annotations_schema_and_boxes(tmp_path: Path):
    """Verify calculated labels, bounding boxes, and exclude flags."""
    ann_data = {
        "labelGroups": [
            {
                "name": "Calculated",
                "labels": [
                    {"id": CALCULATED_LABEL_OPACITY_ID, "name": "Lung Opacity"},
                    {"id": CALCULATED_LABEL_NORMAL_ID, "name": "Normal"},
                    {"id": CALCULATED_LABEL_NOT_NORMAL_ID, "name": "No Lung Opacity / Not Normal"}
                ]
            }
        ],
        "datasets": [
            {
                "annotations": [
                    {  # Lung Opacity with Bounding Box
                        "StudyInstanceUID": "study-1",
                        "labelId": CALCULATED_LABEL_OPACITY_ID,
                        "data": {"x": 100.0, "y": 150.0, "width": 50.0, "height": 80.0}
                    },
                    {  # Second Bounding Box on same study
                        "StudyInstanceUID": "study-1",
                        "labelId": CALCULATED_LABEL_OPACITY_ID,
                        "data": {"x": 200.0, "y": 250.0, "width": 60.0, "height": 90.0}
                    },
                    {  # Normal Study
                        "StudyInstanceUID": "study-2",
                        "labelId": CALCULATED_LABEL_NORMAL_ID,
                        "data": None
                    },
                    {  # Excluded Study (Radiologist Flag)
                        "StudyInstanceUID": "study-3",
                        "labelId": "L_3lj",  # Exclude
                        "data": None
                    }
                ]
            }
        ]
    }
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(ann_data), encoding="utf-8")

    studies, errors = load_rsna_annotations(ann_file)
    assert "study-1" in studies
    assert studies["study-1"]["calculated_label_id"] == CALCULATED_LABEL_OPACITY_ID
    assert len(studies["study-1"]["bounding_boxes"]) == 2
    assert studies["study-1"]["bounding_boxes"][0]["x"] == 100.0

    assert studies["study-2"]["calculated_label_id"] == CALCULATED_LABEL_NORMAL_ID
    assert len(studies["study-2"]["bounding_boxes"]) == 0

    assert studies["study-3"]["has_exclude_flag"] is True


def test_cohort_building_and_exclusion_reasons():
    """Verify classification and exclusion partitioning in cohort builder."""
    mappings = [
        {"SOPInstanceUID": "sop-1", "StudyInstanceUID": "study-1", "SeriesInstanceUID": "ser-1", "img_id": "00000001_000.png", "patient_id": "00000001"},
        {"SOPInstanceUID": "sop-2", "StudyInstanceUID": "study-2", "SeriesInstanceUID": "ser-2", "img_id": "00000002_000.png", "patient_id": "00000002"},
        {"SOPInstanceUID": "sop-3", "StudyInstanceUID": "study-3", "SeriesInstanceUID": "ser-3", "img_id": "00000003_000.png", "patient_id": "00000003"},
        {"SOPInstanceUID": "sop-4", "StudyInstanceUID": "study-4", "SeriesInstanceUID": "ser-4", "img_id": "00000004_000.png", "patient_id": "00000004"},
        {"SOPInstanceUID": "sop-5", "StudyInstanceUID": "study-5", "SeriesInstanceUID": "ser-5", "img_id": "00000005_000.png", "patient_id": "00000005"},
    ]
    annotations = {
        "study-1": {"calculated_label_id": CALCULATED_LABEL_OPACITY_ID, "bounding_boxes": [], "has_exclude_flag": False, "raw_label_ids": []},
        "study-2": {"calculated_label_id": CALCULATED_LABEL_NORMAL_ID, "bounding_boxes": [], "has_exclude_flag": False, "raw_label_ids": []},
        "study-3": {"calculated_label_id": CALCULATED_LABEL_NOT_NORMAL_ID, "bounding_boxes": [], "has_exclude_flag": False, "raw_label_ids": []},
        "study-4": {"calculated_label_id": None, "bounding_boxes": [], "has_exclude_flag": True, "raw_label_ids": ["L_3lj"]},
        # study-5 is missing from annotations entirely
    }

    valid, excluded = build_cohort_records(mappings, annotations)
    assert len(valid) == 3
    assert len(excluded) == 2

    # Check label names
    assert valid[0].label_name == LABEL_LUNG_OPACITY
    assert valid[0].label_index == 1
    assert valid[1].label_name == LABEL_NO_LUNG_OPACITY
    assert valid[1].label_index == 0
    assert valid[2].label_name == LABEL_NO_LUNG_OPACITY
    assert valid[2].label_index == 0

    # Check exclusion reasons
    excluded_by_sop = {ex.sop_instance_uid: ex.reason for ex in excluded}
    assert excluded_by_sop["sop-4"] == "RADIOLOGIST_FLAGGED_EXCLUDE"
    assert excluded_by_sop["sop-5"] == "MISSING_ANNOTATION_RECORD"


# ==============================================================================
# 4. Patient-Level Split & Leakage Tests
# ==============================================================================

def test_patient_level_split_disjointness():
    """Verify zero patient overlap across splits when patients have multiple images."""
    records: List[RSNARecord] = []

    # Create 100 patients with varying image counts (1 to 4 images per patient)
    for p_idx in range(100):
        pid = f"PATIENT_{p_idx:04d}"
        num_images = 1 + (p_idx % 4)
        for img_idx in range(num_images):
            lbl_name = LABEL_LUNG_OPACITY if (p_idx % 3 == 0) else LABEL_NO_LUNG_OPACITY
            records.append(RSNARecord(
                sop_instance_uid=f"sop_{p_idx}_{img_idx}",
                study_instance_uid=f"study_{p_idx}_{img_idx}",
                series_instance_uid="ser",
                nih_image_id=f"{p_idx:08d}_{img_idx:03d}.png",
                patient_id=pid,
                adjudicated_label_id="calc_id",
                raw_class_name="raw",
                label_name=lbl_name,
                label_index=RSNA_LABEL_TO_INT[lbl_name],
                bounding_boxes=[],
                dicom_relative_path=f"sop_{p_idx}_{img_idx}.dcm"
            ))

    splits = create_patient_level_splits(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_pts = {r.patient_id for r in splits["train"]}
    val_pts = {r.patient_id for r in splits["val"]}
    test_pts = {r.patient_id for r in splits["test"]}

    # Critical Assertion: Zero patient overlap across all pairwise combinations
    assert len(train_pts & val_pts) == 0, "Train and Val share patient IDs!"
    assert len(train_pts & test_pts) == 0, "Train and Test share patient IDs!"
    assert len(val_pts & test_pts) == 0, "Val and Test share patient IDs!"

    # Ensure patient proportions match intended ratios
    assert len(train_pts) == 70
    assert len(val_pts) == 15
    assert len(test_pts) == 15

    # Ensure total images match sum of split records
    assert sum(len(r) for r in splits.values()) == len(records)


def test_deterministic_splits_with_fixed_seed():
    """Verify splitting produces identical manifests with same seed and changes with different seed."""
    records = [
        RSNARecord(
            sop_instance_uid=f"sop_{i}",
            study_instance_uid=f"study_{i}",
            series_instance_uid="ser",
            nih_image_id=f"{i:08d}_000.png",
            patient_id=f"PID_{i:04d}",
            adjudicated_label_id="calc",
            raw_class_name="raw",
            label_name=LABEL_NO_LUNG_OPACITY,
            label_index=0,
            bounding_boxes=[],
            dicom_relative_path=f"sop_{i}.dcm"
        )
        for i in range(50)
    ]

    splits_a1 = create_patient_level_splits(records, seed=42)
    splits_a2 = create_patient_level_splits(records, seed=42)
    splits_b = create_patient_level_splits(records, seed=999)

    sops_a1_train = [r.sop_instance_uid for r in splits_a1["train"]]
    sops_a2_train = [r.sop_instance_uid for r in splits_a2["train"]]
    sops_b_train = [r.sop_instance_uid for r in splits_b["train"]]

    assert sops_a1_train == sops_a2_train
    assert sops_a1_train != sops_b_train


# ==============================================================================
# 5. Manifest Serialization & Integrity Tests
# ==============================================================================

def test_save_manifests_content_and_summary(tmp_path: Path):
    """Verify generated CSV manifests and JSON summary contain accurate class counts."""
    r1 = RSNARecord("sop-1", "st-1", "se-1", "00000001_000.png", "00000001", "L_v8n", "Lung Opacity", LABEL_LUNG_OPACITY, 1, [{"x": 10, "y": 20, "width": 30, "height": 40}], "sop-1.dcm", True)
    r2 = RSNARecord("sop-2", "st-2", "se-2", "00000002_000.png", "00000002", "L_o8w", "Normal", LABEL_NO_LUNG_OPACITY, 0, [], "sop-2.dcm", True)
    r3 = RSNARecord("sop-3", "st-3", "se-3", "00000003_000.png", "00000003", "L_yd0", "No Lung Opacity / Not Normal", LABEL_NO_LUNG_OPACITY, 0, [], "sop-3.dcm", True)

    splits = {
        "train": [r1, r2],
        "val": [r3],
        "test": []
    }
    excluded = [
        # RSNAExcludedRecord
        from_src := __import__("src.rsna_dataset", fromlist=["RSNAExcludedRecord"]).RSNAExcludedRecord(
            "sop-ex", "st-ex", "00000004_000.png", "00000004", "RADIOLOGIST_FLAGGED_EXCLUDE", ["L_3lj"]
        )
    ]

    out_dir = tmp_path / "processed_rsna"
    summary = save_manifests(splits, excluded, out_dir, assert_dicoms_exist=False)

    assert (out_dir / "train_manifest.csv").is_file()
    assert (out_dir / "val_manifest.csv").is_file()
    assert (out_dir / "test_manifest.csv").is_file()
    assert (out_dir / "excluded_manifest.csv").is_file()
    assert (out_dir / "split_summary.json").is_file()

    assert summary["splits"]["train"]["class_counts"][LABEL_LUNG_OPACITY] == 1
    assert summary["splits"]["train"]["class_counts"][LABEL_NO_LUNG_OPACITY] == 1
    assert summary["splits"]["val"]["class_counts"][LABEL_NO_LUNG_OPACITY] == 1
    assert summary["patient_leakage_verified_zero"] is True


def test_missing_dicom_assertion_raises_error(tmp_path: Path):
    """Verify assert_dicoms_exist raises FileNotFoundError when a DICOM is missing."""
    r = RSNARecord("sop-missing", "st-1", "se-1", "00000001_000.png", "00000001", "L_o8w", "Normal", LABEL_NO_LUNG_OPACITY, 0, [], "nonexistent.dcm", False)
    splits = {"train": [r], "val": [], "test": []}

    with pytest.raises(FileNotFoundError, match="DICOM for SOP sop-missing missing"):
        save_manifests(splits, [], tmp_path, dicom_root=tmp_path / "nonexistent_dir", assert_dicoms_exist=True)


# ==============================================================================
# 6. DICOM Loading & Preprocessing Tests
# ==============================================================================

def test_load_dicom_monochrome2_normal(tmp_path: Path):
    """Verify reading a standard MONOCHROME2 DICOM file into RGB normalized array."""
    if pydicom is None:
        pytest.skip("pydicom is not installed.")

    # Create synthetic gradient 64x64
    grid = np.linspace(10, 200, 64 * 64, dtype=np.uint8).reshape((64, 64))
    dcm_path = create_minimal_dicom_file(tmp_path / "test_mono2.dcm", grid, photometric="MONOCHROME2")

    arr = load_dicom_as_array(dcm_path, target_size=(224, 224))
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8
    # Min value in grid is mapped to 0, max to 255
    assert arr.min() == 0
    assert arr.max() == 255


def test_load_dicom_monochrome1_inversion(tmp_path: Path):
    """Verify MONOCHROME1 images are inverted so bone/opacity remains bright."""
    if pydicom is None:
        pytest.skip("pydicom is not installed.")

    # In MONOCHROME1, 0 is bright and 255 is dark.
    # An image with a bright spot at (0,0) has value 0 in MONOCHROME1.
    grid = np.zeros((32, 32), dtype=np.uint8)
    grid[0, 0] = 0    # Bright spot in MONOCHROME1
    grid[1:, :] = 200 # Dark background in MONOCHROME1

    dcm_path = create_minimal_dicom_file(tmp_path / "test_mono1.dcm", grid, photometric="MONOCHROME1")
    arr = load_dicom_as_array(dcm_path, target_size=(32, 32))

    # After inversion, value at (0,0) should be the maximum (brightest)
    assert arr[0, 0, 0] == arr.max()


# ==============================================================================
# 7. Streaming Batch Generator Tests
# ==============================================================================

def test_rsna_batch_generator_contract(tmp_path: Path):
    """Verify streaming batch generator yields expected batch dimensions and MobileNetV2 scaling."""
    if pydicom is None:
        pytest.skip("pydicom is not installed.")

    dicom_dir = tmp_path / "dicoms"
    dicom_dir.mkdir()

    # Create 4 synthetic DICOM files
    records = []
    for i in range(4):
        grid = np.full((32, 32), fill_value=50 * (i + 1), dtype=np.uint8)
        sop = f"sop_{i}"
        create_minimal_dicom_file(dicom_dir / f"{sop}.dcm", grid, patient_id=f"PID_{i}")
        lbl_name = LABEL_LUNG_OPACITY if i % 2 == 0 else LABEL_NO_LUNG_OPACITY
        records.append({
            "dicom_relative_path": f"{sop}.dcm",
            "label_index": RSNA_LABEL_TO_INT[lbl_name]
        })

    # Write manifest CSV
    manifest_csv = tmp_path / "train_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dicom_relative_path", "label_index"])
        writer.writeheader()
        writer.writerows(records)

    gen = rsna_batch_generator(manifest_csv, dicom_dir, batch_size=2, target_size=(224, 224), shuffle=False)
    batch_x, batch_y = next(gen)

    assert batch_x.shape == (2, 224, 224, 3)
    assert batch_x.dtype == np.float32
    # Verify MobileNetV2 scaling bounds [-1.0, 1.0]
    assert -1.0 <= batch_x.min() and batch_x.max() <= 1.0
    assert batch_y.shape == (2,)
    assert batch_y.dtype == np.int32
    assert list(batch_y) == [1, 0]


# ==============================================================================
# 8. Pipeline Audit Integration Test
# ==============================================================================

def test_audit_rsna_pipeline_on_project_files():
    """Verify audit_rsna_pipeline passes on the actual project metadata and manifests."""
    from src.config import RSNA_DICOM_DIR, RSNA_METADATA_DIR, RSNA_PROCESSED_DIR

    mapping_file = RSNA_METADATA_DIR / "pneumonia-challenge-dataset-mappings_2018.json"
    if not mapping_file.is_file() or not RSNA_DICOM_DIR.is_dir():
        pytest.skip("RSNA metadata or local DICOM files not present on disk.")

    report = audit_rsna_pipeline(
        metadata_dir=RSNA_METADATA_DIR,
        dicom_dir=RSNA_DICOM_DIR,
        processed_dir=RSNA_PROCESSED_DIR,
        max_readability_checks=50,
    )

    assert report["status"] == "PASS"
    assert report["mapping_count"] == 30000
    assert report["total_valid_cohort"] == 29684
    assert report["total_excluded_cohort"] == 316
    assert report["unique_patients_in_valid_cohort"] == 12233
    assert report["class_distribution"][LABEL_LUNG_OPACITY] == 7106
    assert report["class_distribution"][LABEL_NO_LUNG_OPACITY] == 22578
    assert report["patient_leakage_verified_zero"] is True


def test_build_dicom_index_and_nested_resolution(tmp_path: Path):
    """Verify build_dicom_index and resolve_dicom_path correctly index flat and nested layouts."""
    if pydicom is None:
        pytest.skip("pydicom is not installed.")

    dicom_root = tmp_path / "dicoms"
    dicom_root.mkdir()

    # 1. Flat layout
    grid = np.zeros((16, 16), dtype=np.uint8)
    create_minimal_dicom_file(dicom_root / "sop_flat.dcm", grid)

    # 2. Nested layout: <study>/<series>/<sop>.dcm
    nested_dir = dicom_root / "study_123" / "series_456"
    create_minimal_dicom_file(nested_dir / "sop_nested.dcm", grid)

    # 3. Archive root layout: mdai_rsna_.../<study>/<series>/<sop>.dcm
    archive_dir = dicom_root / "mdai_rsna_project_x9N20BZa_images_2018-07-20-153330" / "study_789" / "series_012"
    create_minimal_dicom_file(archive_dir / "sop_archive.dcm", grid)

    # Test indexing
    index = build_dicom_index(dicom_root)
    assert len(index) == 3
    assert "sop_flat" in index
    assert "sop_nested" in index
    assert "sop_archive" in index

    # Test resolution with index
    p_flat, rel_flat = resolve_dicom_path("sop_flat", dicom_root, dicom_index=index)
    assert p_flat is not None and p_flat.is_file()
    assert rel_flat == "sop_flat.dcm"

    p_nested, rel_nested = resolve_dicom_path(
        "sop_nested", dicom_root, "study_123", "series_456", dicom_index=index
    )
    assert p_nested is not None and p_nested.is_file()
    assert "study_123" in rel_nested

    # Test resolution without index (fallback paths)
    p_arch, rel_arch = resolve_dicom_path(
        "sop_archive", dicom_root, "study_789", "series_012", dicom_index=None
    )
    assert p_arch is not None and p_arch.is_file()
    assert "sop_archive.dcm" in rel_arch


def test_validate_local_dicoms_healthy_and_corrupt(tmp_path: Path):
    """Verify validate_local_dicoms distinguishes readable DICOMs from corrupt or empty files."""
    if pydicom is None:
        pytest.skip("pydicom is not installed.")

    dicom_dir = tmp_path / "dicoms_val"
    dicom_dir.mkdir()

    grid = np.ones((16, 16), dtype=np.uint8) * 100
    valid_dcm = create_minimal_dicom_file(dicom_dir / "valid.dcm", grid)

    # Corrupt 0-byte file
    zero_dcm = dicom_dir / "zero.dcm"
    zero_dcm.write_bytes(b"")

    # Corrupt truncated header
    bad_dcm = dicom_dir / "bad.dcm"
    bad_dcm.write_bytes(b"DICM_TRUNCATED_NOT_VALID")

    res = validate_local_dicoms([valid_dcm, zero_dcm, bad_dcm])
    assert res["checked_count"] == 3
    assert res["valid_count"] == 1
    assert res["corrupt_count"] == 2
    corrupt_names = [Path(p).name for p, _ in res["corrupt_files"]]
    assert "zero.dcm" in corrupt_names
    assert "bad.dcm" in corrupt_names


def test_audit_rsna_pipeline_multi_tier_statuses():
    """Verify audit_rsna_pipeline returns structured 5-tier status reports accurately."""
    from src.config import RSNA_DICOM_DIR, RSNA_METADATA_DIR, RSNA_PROCESSED_DIR

    mapping_file = RSNA_METADATA_DIR / "pneumonia-challenge-dataset-mappings_2018.json"
    if not mapping_file.is_file() or not RSNA_DICOM_DIR.is_dir():
        pytest.skip("RSNA metadata or local DICOM files not present on disk.")

    report = audit_rsna_pipeline(
        metadata_dir=RSNA_METADATA_DIR,
        dicom_dir=RSNA_DICOM_DIR,
        processed_dir=RSNA_PROCESSED_DIR,
        max_readability_checks=50,
    )

    # Tier 1
    assert report["metadata_status"] == "PASS"

    # Tier 2
    assert report["manifest_status"] == "PASS"
    assert report["patient_leakage_verified_zero"] is True

    # Tier 3 (complete local dicoms present)
    assert report["completeness_status"] == "COMPLETE"
    assert report["local_dicoms_present_count"] >= 29684

    # Tier 4 (readability validation checked)
    assert "VERIFIED" in report["readability_status"]
    assert report["readability_validation"]["valid_count"] == 50
    assert report["readability_validation"]["corrupt_count"] == 0

    # Tier 5 (overall readiness is READY with complete cohort)
    assert report["overall_readiness"] == "READY"
    assert report["is_ready_for_training"] is True

