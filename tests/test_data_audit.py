"""Unit tests for the dataset auditing and validation module.

Verifies directory scanning, split detection, per-class image counts,
corrupt image detection, duplicate image detection, warning generations,
and formatting functions on both empty and mock datasets.
"""

from pathlib import Path
from typing import Dict

import pytest
from PIL import Image

from src.data_audit import audit_dataset, format_audit_summary


def test_audit_empty_or_nonexistent_directory(tmp_path: Path) -> None:
    """Verify auditing handles missing directories gracefully without raising exceptions."""
    non_existent = tmp_path / "does_not_exist"
    report = audit_dataset(non_existent)

    assert isinstance(report, dict)
    assert report["exists"] is False
    assert report["is_ready_for_training"] is False
    assert len(report["warnings"]) > 0

    summary_text = format_audit_summary(report)
    assert "DATASET AUDIT REPORT" in summary_text
    assert "Directory Exists: False" in summary_text


def test_audit_populated_directory_structure(tmp_path: Path) -> None:
    """Verify auditing detects splits, classes, and valid image counts correctly."""
    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    sample_img = Image.new("RGB", (20, 20), color=(128, 128, 128))

    for split in splits:
        for idx, cls_name in enumerate(classes):
            folder = tmp_path / split / cls_name
            folder.mkdir(parents=True)
            # Create distinct valid images to avoid duplicate hash detection
            img1 = Image.new("RGB", (20, 20), color=(idx * 10 + 1, 0, 0))
            img2 = Image.new("RGB", (20, 20), color=(idx * 10 + 2, 0, 0))
            img1.save(folder / "img1.jpeg")
            img2.save(folder / "img2.png")
            # Create an unsupported file that should be ignored
            (folder / "notes.txt").write_text("ignore me")

    report = audit_dataset(tmp_path)

    assert report["exists"] is True
    assert set(report["splits_found"]) == set(splits)
    assert len(report["missing_splits"]) == 0
    # 3 splits * 2 classes * 2 supported images = 12 images
    assert report["total_images"] == 12
    assert len(report["corrupt_files"]) == 0
    assert report["is_ready_for_training"] is True

    # Validate split details
    for split in splits:
        assert report["split_details"][split]["total_images"] == 4
        assert report["split_details"][split]["classes"]["NORMAL"]["image_count"] == 2
        assert report["split_details"][split]["classes"]["PNEUMONIA"]["image_count"] == 2


def test_audit_flags_missing_classes_and_unexpected_items(tmp_path: Path) -> None:
    """Verify audit flags missing expected class and unexpected items."""
    train_dir = tmp_path / "train"
    (train_dir / "NORMAL").mkdir(parents=True)
    img = Image.new("RGB", (10, 10), color="white")
    img.save(train_dir / "NORMAL" / "sample.jpeg")

    (train_dir / "COVID").mkdir(parents=True)
    img.save(train_dir / "COVID" / "sample.jpeg")

    report = audit_dataset(tmp_path, expected_splits=("train",))

    assert "PNEUMONIA" in report["split_details"]["train"]["missing_classes"]
    assert "COVID" in report["split_details"]["train"]["unexpected_items"]
    assert len(report["warnings"]) > 0


def test_audit_detects_corrupt_and_duplicate_images(tmp_path: Path) -> None:
    """Verify audit flags empty/corrupt files and identical duplicate content."""
    train_dir = tmp_path / "train"
    normal_dir = train_dir / "NORMAL"
    pneumonia_dir = train_dir / "PNEUMONIA"
    normal_dir.mkdir(parents=True)
    pneumonia_dir.mkdir(parents=True)

    # 1. Valid image
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(normal_dir / "valid_image.png")

    # 2. Corrupt/empty image (0 bytes)
    (normal_dir / "corrupt_zero_bytes.jpeg").write_bytes(b"")

    # 3. Corrupt truncated header
    (normal_dir / "corrupt_header.png").write_bytes(b"not_a_valid_png_content")

    # 4. Duplicate image across classes
    img.save(pneumonia_dir / "duplicate_of_valid.png")

    report = audit_dataset(tmp_path, expected_splits=("train",))

    assert len(report["corrupt_files"]) == 2
    assert len(report["duplicate_files"]) >= 1
    assert report["is_ready_for_training"] is False  # Corrupt images block training readiness
    assert any("corrupt/empty" in w for w in report["warnings"])
    assert any("duplicate image" in w for w in report["warnings"])
