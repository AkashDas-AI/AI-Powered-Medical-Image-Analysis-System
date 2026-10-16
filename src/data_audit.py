"""Dataset structure auditing and validation module.

Inspects raw and processed dataset directories, verifies clinical split layouts,
counts supported medical image formats, and flags anomalies or missing classes
without blocking runtime execution if the dataset is absent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import CLASS_NAMES, RAW_DATA_DIR, SUPPORTED_EXTENSIONS


def audit_dataset(
    data_dir: Optional[Union[str, Path]] = None,
    expected_splits: Tuple[str, ...] = ("train", "val", "test"),
    expected_classes: Tuple[str, ...] = CLASS_NAMES,
    supported_extensions: Tuple[str, ...] = SUPPORTED_EXTENSIONS,
) -> Dict[str, Any]:
    """Inspect and audit the medical image dataset directory.

    Checks:
    - Existence of root data directory.
    - Presence of expected data splits (e.g., train, val, test).
    - Presence of expected class subdirectories (e.g., NORMAL, PNEUMONIA).
    - Per-class image counts for supported extensions.
    - Identification of unexpected directories or invalid files.

    Args:
        data_dir: Root path to inspect. Defaults to RAW_DATA_DIR.
        expected_splits: Splits to search for. Defaults to ('train', 'val', 'test').
        expected_classes: Class directory names expected. Defaults to ('NORMAL', 'PNEUMONIA').
        supported_extensions: Image extensions to count. Defaults to config extensions.

    Returns:
        Dict[str, Any]: Structured audit report detailing counts, status, and warnings.
    """
    root_path = Path(data_dir) if data_dir is not None else RAW_DATA_DIR

    # Initialize structured report
    report: Dict[str, Any] = {
        "root_path": str(root_path),
        "exists": root_path.exists(),
        "is_directory": root_path.is_dir() if root_path.exists() else False,
        "resolved_data_root": None,
        "splits_found": [],
        "missing_splits": [],
        "split_details": {},
        "total_images": 0,
        "corrupt_files": [],
        "duplicate_files": [],
        "warnings": [],
        "is_ready_for_training": False,
    }
    seen_hashes: Dict[str, str] = {}

    if not report["exists"] or not report["is_directory"]:
        report["warnings"].append(
            f"Dataset directory not found or not a directory: {root_path}. "
            "Please download the dataset per instructions in data/README.md."
        )
        return report

    # Check whether dataset is directly under root or nested under 'chest_xray'
    target_root = root_path
    nested_chest_xray = root_path / "chest_xray"
    if nested_chest_xray.is_dir():
        target_root = nested_chest_xray

    report["resolved_data_root"] = str(target_root)

    # Inspect splits
    for split_name in expected_splits:
        split_path = target_root / split_name
        split_info: Dict[str, Any] = {
            "path": str(split_path),
            "exists": split_path.is_dir(),
            "classes": {},
            "total_images": 0,
            "missing_classes": [],
            "unexpected_items": [],
        }

        if not split_path.is_dir():
            report["missing_splits"].append(split_name)
            report["split_details"][split_name] = split_info
            continue

        report["splits_found"].append(split_name)

        # Inspect classes within the split
        found_class_dirs: List[str] = []
        for child in split_path.iterdir():
            if child.is_dir():
                found_class_dirs.append(child.name)
                # Count valid images
                valid_images: List[Path] = []
                corrupt_images: List[str] = []

                for f in child.iterdir():
                    if f.is_file() and f.suffix.lower() in supported_extensions:
                        # Lightweight corruption check: non-zero size & valid image header
                        if f.stat().st_size == 0:
                            corrupt_images.append(str(f))
                            continue
                        try:
                            from PIL import Image
                            with Image.open(f) as img:
                                img.verify()
                            valid_images.append(f)
                        except Exception:
                            corrupt_images.append(str(f))

                count = len(valid_images)
                split_info["classes"][child.name] = {
                    "image_count": count,
                    "is_expected": child.name in expected_classes,
                    "corrupt_count": len(corrupt_images),
                }
                split_info["total_images"] += count
                report["total_images"] += count

                if corrupt_images:
                    report["corrupt_files"].extend(corrupt_images)
                    report["warnings"].append(
                        f"Found {len(corrupt_images)} corrupt/empty image(s) in {split_name}/{child.name}."
                    )

                if child.name not in expected_classes:
                    split_info["unexpected_items"].append(child.name)

                # Check duplicates via content hash
                import hashlib
                for f in valid_images:
                    try:
                        file_hash = hashlib.md5(f.read_bytes()).hexdigest()
                        if file_hash in seen_hashes:
                            report["duplicate_files"].append((str(f), seen_hashes[file_hash]))
                        else:
                            seen_hashes[file_hash] = str(f)
                    except Exception:
                        pass
            else:
                split_info["unexpected_items"].append(child.name)

        # Identify missing classes
        for expected in expected_classes:
            if expected not in found_class_dirs:
                split_info["missing_classes"].append(expected)

        report["split_details"][split_name] = split_info

    # Compile warnings
    if report["duplicate_files"]:
        report["warnings"].append(
            f"Found {len(report['duplicate_files'])} duplicate image file(s) across dataset."
        )

    if report["missing_splits"]:
        report["warnings"].append(
            f"Missing expected splits: {report['missing_splits']}"
        )

    for split_name, details in report["split_details"].items():
        if details["exists"]:
            if details["missing_classes"]:
                report["warnings"].append(
                    f"Split '{split_name}' is missing expected class directories: {details['missing_classes']}"
                )
            if details["unexpected_items"]:
                report["warnings"].append(
                    f"Split '{split_name}' contains unexpected items: {details['unexpected_items']}"
                )
            if details["total_images"] == 0:
                report["warnings"].append(
                    f"Split '{split_name}' contains 0 supported images."
                )

    # Determine readiness
    report["is_ready_for_training"] = (
        len(report["splits_found"]) >= 2
        and report["total_images"] > 0
        and len(report["missing_splits"]) == 0
        and len(report["corrupt_files"]) == 0
    )

    return report


def format_audit_summary(report: Dict[str, Any]) -> str:
    """Format an audit report dictionary into a readable CLI summary string.

    Args:
        report: Dictionary generated by audit_dataset().

    Returns:
        str: Multiline formatted report.
    """
    lines = [
        "=" * 60,
        "DATASET AUDIT REPORT",
        "=" * 60,
        f"Target Directory: {report['root_path']}",
        f"Resolved Data Root: {report.get('resolved_data_root') or 'N/A'}",
        f"Directory Exists: {report['exists']}",
        f"Total Supported Images Found: {report['total_images']}",
        f"Corrupt/Unreadable Images: {len(report.get('corrupt_files', []))}",
        f"Duplicate Images Detected: {len(report.get('duplicate_files', []))}",
        f"Splits Found: {report['splits_found']}",
        f"Missing Splits: {report['missing_splits']}",
        "-" * 60,
    ]

    for split_name, details in report.get("split_details", {}).items():
        lines.append(f"Split: [{split_name.upper()}] (Exists: {details['exists']})")
        if details["exists"]:
            for cls_name, cls_data in details.get("classes", {}).items():
                expected_tag = "" if cls_data["is_expected"] else " [UNEXPECTED]"
                lines.append(f"  └── {cls_name}: {cls_data['image_count']} images{expected_tag}")
            if details.get("missing_classes"):
                lines.append(f"  [!] Missing Classes: {details['missing_classes']}")
            if details.get("unexpected_items"):
                lines.append(f"  [!] Unexpected Items: {details['unexpected_items']}")

    lines.append("-" * 60)
    if report.get("warnings"):
        lines.append("Warnings / Findings:")
        for w in report["warnings"]:
            lines.append(f"  * {w}")
    else:
        lines.append("No configuration warnings detected.")

    lines.append("-" * 60)
    lines.append(
        f"Ready for Training: {'YES [READY]' if report['is_ready_for_training'] else 'NO [REQUIRES DATASET SETUP]'}"
    )
    lines.append("=" * 60)

    return "\n".join(lines)

