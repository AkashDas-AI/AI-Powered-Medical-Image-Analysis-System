"""RSNA Pneumonia Detection Challenge (2018) Dataset Module.

This module provides an annotation-aware, memory-efficient data pipeline for the
RSNA 2018 challenge dataset:
- Annotation schema parsing (adjudicated consensus classes).
- Standardized binary label mapping:
    - 'LUNG_OPACITY': Adjudicated Lung Opacity (positive class, index 1).
    - 'NO_LUNG_OPACITY': Adjudicated Normal or No Lung Opacity / Not Normal (negative class, index 0).
- Explicit exclusion of uncalculated, ambiguous, or radiologist-flagged exclusion cases.
- Leak-free patient-level train/validation/test splitting derived from official NIH CXR8 mapping.
- Memory-efficient streaming DICOM loading compatible with constrained host memory (<1.0 GB RAM).
- Manifest generation and dataset auditing.

Attribution & Provenance:
- Data Provider: NIH Clinical Center (CVPR 2017) & RSNA / STR (Radiology: AI 2019).
- Educational research prototype: Not intended for clinical diagnostic decision-making.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# --- Standardized Label Constants ---
LABEL_LUNG_OPACITY = "LUNG_OPACITY"
LABEL_NO_LUNG_OPACITY = "NO_LUNG_OPACITY"

RSNA_CLASS_NAMES = [LABEL_NO_LUNG_OPACITY, LABEL_LUNG_OPACITY]
RSNA_LABEL_TO_INT = {LABEL_NO_LUNG_OPACITY: 0, LABEL_LUNG_OPACITY: 1}
RSNA_INT_TO_LABEL = {0: LABEL_NO_LUNG_OPACITY, 1: LABEL_LUNG_OPACITY}

# Adjudicated Calculated Label IDs from md.ai schema
CALCULATED_LABEL_OPACITY_ID = "L_v8n"       # Lung Opacity
CALCULATED_LABEL_NORMAL_ID = "L_o8w"        # Normal
CALCULATED_LABEL_NOT_NORMAL_ID = "L_yd0"    # No Lung Opacity / Not Normal

SUPPORTED_CALCULATED_LABELS = {
    CALCULATED_LABEL_OPACITY_ID: (LABEL_LUNG_OPACITY, "Lung Opacity"),
    CALCULATED_LABEL_NORMAL_ID: (LABEL_NO_LUNG_OPACITY, "Normal"),
    CALCULATED_LABEL_NOT_NORMAL_ID: (LABEL_NO_LUNG_OPACITY, "No Lung Opacity / Not Normal"),
}

# Exclusion categories for radiologist flags
EXCLUDE_LABEL_IDS = {"L_3lj", "L_z94", "L_kx3"}


@dataclass
class RSNARecord:
    """Represents a validated RSNA study record with patient metadata and labels."""
    sop_instance_uid: str
    study_instance_uid: str
    series_instance_uid: str
    nih_image_id: str
    patient_id: str
    adjudicated_label_id: str
    raw_class_name: str
    label_name: str
    label_index: int
    bounding_boxes: List[Dict[str, float]]
    dicom_relative_path: str
    dicom_exists_locally: bool = False


@dataclass
class RSNAExcludedRecord:
    """Represents a record excluded from model training/evaluation with reason."""
    sop_instance_uid: str
    study_instance_uid: str
    nih_image_id: str
    patient_id: str
    reason: str
    raw_labels: List[str]


def extract_patient_id(nih_image_id: str) -> Optional[str]:
    """Extract 8-digit patient ID from NIH CXR8 filename format (e.g. '00012773_000.png')."""
    if not nih_image_id or not isinstance(nih_image_id, str):
        return None
    match = re.match(r"^(\d+)_", nih_image_id.strip())
    return match.group(1) if match else None


def load_rsna_mapping(mapping_path: Path | str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load and validate the official RSNA-to-NIH dataset mapping file.

    Returns:
        tuple of (valid_mappings, mapping_errors_by_sop)
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"RSNA mapping file not found at: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected mapping JSON to contain a list, got {type(data)}")

    valid_mappings: List[Dict[str, Any]] = []
    seen_sops: Set[str] = set()
    errors: Dict[str, str] = {}

    required_fields = ["SOPInstanceUID", "StudyInstanceUID", "img_id"]

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors[f"index_{idx}"] = "Item is not a dictionary"
            continue

        missing = [f for f in required_fields if f not in item or not item[f]]
        if missing:
            sop = item.get("SOPInstanceUID", f"index_{idx}")
            errors[sop] = f"Missing required fields: {missing}"
            continue

        sop = item["SOPInstanceUID"].strip()
        if sop in seen_sops:
            errors[sop] = "Duplicate SOPInstanceUID in mapping"
            continue

        patient_id = extract_patient_id(item["img_id"])
        if not patient_id:
            errors[sop] = f"Could not parse valid NIH patient ID from img_id '{item['img_id']}'"
            continue

        item_copy = dict(item)
        item_copy["patient_id"] = patient_id
        seen_sops.add(sop)
        valid_mappings.append(item_copy)

    return valid_mappings, errors


def load_rsna_annotations(
    annotations_path: Path | str
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Load and index adjudicated annotations by StudyInstanceUID.

    Returns:
        tuple of (study_annotations_dict, error_dict)
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        raise FileNotFoundError(f"RSNA annotations file not found at: {annotations_path}")

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "datasets" not in data or not data["datasets"]:
        raise ValueError("Invalid md.ai annotations JSON schema: missing 'datasets'")

    dataset = data["datasets"][0]
    annotations_list = dataset.get("annotations", [])
    if not isinstance(annotations_list, list):
        raise ValueError("Annotations payload is not a list")

    # Map label IDs to descriptions
    label_names = {}
    for lg in data.get("labelGroups", []):
        for l in lg.get("labels", []):
            label_names[l["id"]] = f"{lg.get('name')}: {l.get('name')}"

    # Group annotations by StudyInstanceUID
    studies_annotations: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    for a in annotations_list:
        study_uid = a.get("StudyInstanceUID")
        if not study_uid:
            continue

        if study_uid not in studies_annotations:
            studies_annotations[study_uid] = {
                "calculated_label_id": None,
                "calculated_raw_name": None,
                "bounding_boxes": [],
                "raw_label_ids": [],
                "has_exclude_flag": False,
            }

        rec = studies_annotations[study_uid]
        lid = a.get("labelId")
        rec["raw_label_ids"].append(lid)

        if lid in EXCLUDE_LABEL_IDS:
            rec["has_exclude_flag"] = True

        if lid in SUPPORTED_CALCULATED_LABELS:
            rec["calculated_label_id"] = lid
            rec["calculated_raw_name"] = SUPPORTED_CALCULATED_LABELS[lid][1]
            if lid == CALCULATED_LABEL_OPACITY_ID and a.get("data"):
                # Save bounding box coordinates
                bbox = a["data"]
                if isinstance(bbox, dict) and "x" in bbox and "y" in bbox:
                    rec["bounding_boxes"].append({
                        "x": float(bbox.get("x", 0)),
                        "y": float(bbox.get("y", 0)),
                        "width": float(bbox.get("width", 0)),
                        "height": float(bbox.get("height", 0)),
                    })

    return studies_annotations, errors


def build_dicom_index(dicom_root: Path | str) -> Dict[str, Path]:
    """Scan dicom_root and build an in-memory map of SOPInstanceUID -> relative Path.

    Handles flat layout (<sop>.dcm), nested (<study>/<series>/<sop>.dcm),
    and archive nested (mdai_rsna_.../<study>/<series>/<sop>.dcm).
    """
    root = Path(dicom_root)
    index: Dict[str, Path] = {}
    if not root.is_dir():
        return index

    for entry in root.rglob("*.dcm"):
        if entry.is_file():
            sop_uid = entry.stem  # Filename without .dcm is the SOPInstanceUID
            try:
                rel = entry.relative_to(root)
                index[sop_uid] = rel
            except ValueError:
                pass
    return index


def resolve_dicom_path(
    sop_instance_uid: str,
    dicom_root: Path | str,
    study_instance_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    dicom_index: Optional[Dict[str, Path]] = None,
) -> Tuple[Optional[Path], str]:
    """Locate DICOM file under root in either flat or nested directory structures.

    Returns:
        tuple of (absolute_path_if_exists, relative_path_string)
    """
    root = Path(dicom_root)

    # 1. Fast O(1) index lookup if index provided
    if dicom_index is not None and sop_instance_uid in dicom_index:
        rel = dicom_index[sop_instance_uid]
        return root / rel, str(rel).replace("\\", "/")

    # 2. Check flat path
    flat_rel = f"{sop_instance_uid}.dcm"
    flat_path = root / flat_rel
    if flat_path.is_file():
        return flat_path, flat_rel

    # 3. Check standard study/series nested path
    if study_instance_uid and series_instance_uid:
        nested_rel = f"{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.dcm"
        nested_path = root / nested_rel
        if nested_path.is_file():
            return nested_path, nested_rel

        # 4. Check archive root prefix nested path
        archive_prefix_rel = f"mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.dcm"
        archive_path = root / archive_prefix_rel
        if archive_path.is_file():
            return archive_path, archive_prefix_rel

    # Default expected relative path is flat
    return None, flat_rel


def build_cohort_records(
    mappings: List[Dict[str, Any]],
    studies_annotations: Dict[str, Dict[str, Any]],
    dicom_dir: Optional[Path | str] = None,
) -> Tuple[List[RSNARecord], List[RSNAExcludedRecord]]:
    """Merge mapping metadata and annotations into classified and excluded cohorts."""
    valid_records: List[RSNARecord] = []
    excluded_records: List[RSNAExcludedRecord] = []

    # Pre-build DICOM index once to ensure O(1) lookups across all 30,000 records
    dicom_index = build_dicom_index(dicom_dir) if dicom_dir else None

    for m in mappings:
        sop = m["SOPInstanceUID"]
        study = m["StudyInstanceUID"]
        series = m.get("SeriesInstanceUID", "")
        nih_id = m["img_id"]
        patient_id = m["patient_id"]

        ann = studies_annotations.get(study)
        if not ann:
            excluded_records.append(RSNAExcludedRecord(
                sop_instance_uid=sop,
                study_instance_uid=study,
                nih_image_id=nih_id,
                patient_id=patient_id,
                reason="MISSING_ANNOTATION_RECORD",
                raw_labels=[]
            ))
            continue

        if ann["has_exclude_flag"]:
            excluded_records.append(RSNAExcludedRecord(
                sop_instance_uid=sop,
                study_instance_uid=study,
                nih_image_id=nih_id,
                patient_id=patient_id,
                reason="RADIOLOGIST_FLAGGED_EXCLUDE",
                raw_labels=ann["raw_label_ids"]
            ))
            continue

        calc_id = ann["calculated_label_id"]
        if not calc_id or calc_id not in SUPPORTED_CALCULATED_LABELS:
            excluded_records.append(RSNAExcludedRecord(
                sop_instance_uid=sop,
                study_instance_uid=study,
                nih_image_id=nih_id,
                patient_id=patient_id,
                reason="UNRESOLVED_CONSENSUS_OR_UNSUPPORTED",
                raw_labels=ann["raw_label_ids"]
            ))
            continue

        label_name, raw_name = SUPPORTED_CALCULATED_LABELS[calc_id]
        label_idx = RSNA_LABEL_TO_INT[label_name]

        # Resolve DICOM path if directory provided
        dicom_exists = False
        dicom_rel = f"{sop}.dcm"
        if dicom_dir:
            resolved_p, rel = resolve_dicom_path(
                sop, dicom_dir, study, series, dicom_index=dicom_index
            )
            dicom_rel = rel
            dicom_exists = resolved_p is not None

        valid_records.append(RSNARecord(
            sop_instance_uid=sop,
            study_instance_uid=study,
            series_instance_uid=series,
            nih_image_id=nih_id,
            patient_id=patient_id,
            adjudicated_label_id=calc_id,
            raw_class_name=raw_name,
            label_name=label_name,
            label_index=label_idx,
            bounding_boxes=ann["bounding_boxes"],
            dicom_relative_path=dicom_rel,
            dicom_exists_locally=dicom_exists
        ))

    return valid_records, excluded_records


def create_patient_level_splits(
    records: List[RSNARecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Dict[str, List[RSNARecord]]:
    """Partition records into train/val/test splits strictly grouped by patient ID.

    Ensures zero patient leakage across splits:
        set(Train_Patients) ∩ set(Val_Patients) = ∅
        set(Train_Patients) ∩ set(Test_Patients) = ∅
        set(Val_Patients) ∩ set(Test_Patients) = ∅

    Target proportions are applied directly to the unique patient population.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    if not records:
        return {"train": [], "val": [], "test": []}

    # Group records by patient ID
    patient_to_records: Dict[str, List[RSNARecord]] = {}
    for r in records:
        patient_to_records.setdefault(r.patient_id, []).append(r)

    # Sort patient IDs for deterministic ordering before permutation
    unique_patients = sorted(patient_to_records.keys())
    num_patients = len(unique_patients)

    # Deterministic permutation using fixed seed
    rng = np.random.default_rng(seed)
    shuffled_patients = list(unique_patients)
    rng.shuffle(shuffled_patients)

    # Calculate exact patient cutoff indices
    train_end = int(round(num_patients * train_ratio))
    val_end = train_end + int(round(num_patients * val_ratio))

    train_patients = set(shuffled_patients[:train_end])
    val_patients = set(shuffled_patients[train_end:val_end])
    test_patients = set(shuffled_patients[val_end:])

    # Programmatic assertion of patient disjointness
    assert len(train_patients & val_patients) == 0, "Patient leakage between train and val!"
    assert len(train_patients & test_patients) == 0, "Patient leakage between train and test!"
    assert len(val_patients & test_patients) == 0, "Patient leakage between val and test!"

    splits: Dict[str, List[RSNARecord]] = {"train": [], "val": [], "test": []}
    for pid, r_list in patient_to_records.items():
        if pid in train_patients:
            splits["train"].extend(r_list)
        elif pid in val_patients:
            splits["val"].extend(r_list)
        elif pid in test_patients:
            splits["test"].extend(r_list)
        else:
            raise RuntimeError(f"Patient {pid} was not assigned to any split")

    return splits


def save_manifests(
    splits: Dict[str, List[RSNARecord]],
    excluded: List[RSNAExcludedRecord],
    output_dir: Path | str,
    dicom_root: Optional[Path | str] = None,
    assert_dicoms_exist: bool = False
) -> Dict[str, Any]:
    """Serialize split manifests and summary metadata into CSV and JSON files under output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sop_instance_uid",
        "study_instance_uid",
        "nih_image_id",
        "patient_id",
        "label_name",
        "label_index",
        "raw_class_name",
        "dicom_relative_path",
        "dicom_exists_locally",
        "num_bounding_boxes",
        "bounding_boxes_json"
    ]

    summary: Dict[str, Any] = {
        "dataset_name": "RSNA Pneumonia Detection Challenge (2018)",
        "task_name": "binary_lung_opacity_classification",
        "class_mapping": RSNA_LABEL_TO_INT,
        "splits": {},
        "excluded_count": len(excluded),
        "total_classified_images": sum(len(records) for records in splits.values()),
        "patient_leakage_verified_zero": True,
    }

    patient_sets: Dict[str, Set[str]] = {}

    for split_name, records in splits.items():
        csv_file = out_path / f"{split_name}_manifest.csv"
        pts: Set[str] = set()
        class_counts: Dict[str, int] = {LABEL_NO_LUNG_OPACITY: 0, LABEL_LUNG_OPACITY: 0}

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                pts.add(r.patient_id)
                class_counts[r.label_name] = class_counts.get(r.label_name, 0) + 1

                if assert_dicoms_exist and dicom_root:
                    full_p = Path(dicom_root) / r.dicom_relative_path
                    if not full_p.is_file():
                        raise FileNotFoundError(
                            f"Assertion failed: DICOM for SOP {r.sop_instance_uid} missing at {full_p}"
                        )

                writer.writerow({
                    "sop_instance_uid": r.sop_instance_uid,
                    "study_instance_uid": r.study_instance_uid,
                    "nih_image_id": r.nih_image_id,
                    "patient_id": r.patient_id,
                    "label_name": r.label_name,
                    "label_index": r.label_index,
                    "raw_class_name": r.raw_class_name,
                    "dicom_relative_path": r.dicom_relative_path,
                    "dicom_exists_locally": r.dicom_exists_locally,
                    "num_bounding_boxes": len(r.bounding_boxes),
                    "bounding_boxes_json": json.dumps(r.bounding_boxes)
                })

        patient_sets[split_name] = pts
        summary["splits"][split_name] = {
            "num_images": len(records),
            "num_unique_patients": len(pts),
            "class_counts": class_counts,
            "manifest_file": str(csv_file.name)
        }

    # Verify cross-split disjointness
    summary["patient_leakage_verified_zero"] = (
        len(patient_sets.get("train", set()) & patient_sets.get("val", set())) == 0 and
        len(patient_sets.get("train", set()) & patient_sets.get("test", set())) == 0 and
        len(patient_sets.get("val", set()) & patient_sets.get("test", set())) == 0
    )

    # Save excluded manifest
    excluded_csv = out_path / "excluded_manifest.csv"
    with open(excluded_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sop_instance_uid", "study_instance_uid", "nih_image_id", "patient_id", "reason", "raw_labels"])
        writer.writeheader()
        for ex in excluded:
            writer.writerow({
                "sop_instance_uid": ex.sop_instance_uid,
                "study_instance_uid": ex.study_instance_uid,
                "nih_image_id": ex.nih_image_id,
                "patient_id": ex.patient_id,
                "reason": ex.reason,
                "raw_labels": ",".join(ex.raw_labels)
            })

    # Save split summary JSON
    summary_file = out_path / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_dicom_as_array(
    dicom_path: Path | str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """Read a DICOM file, apply windowing/photometric inversion, and return a (target_size, 3) uint8 array.

    Memory-efficient: processes a single image at a time with minimal temporary allocations.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to decode DICOM images. Run: pip install pydicom")

    dcm_path = Path(dicom_path)
    if not dcm_path.is_file():
        raise FileNotFoundError(f"DICOM file does not exist: {dcm_path}")

    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise ValueError(
            f"Failed to read or decode DICOM image from {dcm_path.name}: {type(exc).__name__}"
        ) from exc

    if arr.size == 0 or arr.ndim != 2:
        raise ValueError(
            f"Invalid DICOM pixel array dimensions {arr.shape} in {dcm_path.name}; expected 2D radiograph."
        )

    # Photometric Interpretation handling: MONOCHROME1 has 0 as bright, needs inversion
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    # Windowing / normalization to 0..255
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)

    arr_uint8 = arr.astype(np.uint8)

    # Convert to PIL Image for standardized RGB resizing
    pil_img = Image.fromarray(arr_uint8).convert("RGB")
    expected_pil_size = (target_size[1], target_size[0])
    if pil_img.size != expected_pil_size:
        pil_img = pil_img.resize(expected_pil_size, Image.Resampling.BILINEAR)

    return np.array(pil_img, dtype=np.uint8)


# Try importing Keras PyDataset / Sequence base class safely
try:
    import keras
    if hasattr(keras.utils, "PyDataset"):
        _BaseDataset = keras.utils.PyDataset
    elif hasattr(keras.utils, "Sequence"):
        _BaseDataset = keras.utils.Sequence
    else:
        _BaseDataset = object
except ImportError:
    try:
        from tensorflow.keras.utils import Sequence as _BaseDataset  # type: ignore
    except ImportError:
        _BaseDataset = object


class RSNABatchGenerator(_BaseDataset):
    """Memory-efficient, bounded batch generator for streaming RSNA DICOM images.

    Complies with Keras PyDataset / Sequence interface and generator protocol.
    Streams DICOM files on-the-fly, resizes to target_size, and scales pixels to
    [-1.0, 1.0] for MobileNetV2 without buffering the whole dataset into RAM.
    """

    def __init__(
        self,
        manifest_csv: Path | str,
        dicom_root: Path | str,
        batch_size: int = 8,
        target_size: Tuple[int, int] = (224, 224),
        shuffle: bool = True,
        seed: int = RANDOM_SEED,
        class_weights: Optional[Dict[int, float]] = None,
        max_samples: Optional[int] = None,
        num_workers: int = 4,
        **kwargs: Any,
    ):
        if _BaseDataset is not object:
            super().__init__(**kwargs)
        self.manifest_path = Path(manifest_csv)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.dicom_root = Path(dicom_root)
        self.batch_size = max(1, int(batch_size))
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed
        self.class_weights = class_weights
        self.num_workers = max(1, int(num_workers))

        # Ingest manifest records (lightweight relative paths and labels)
        self.entries: List[Tuple[str, int, str]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["dicom_relative_path"]
                label_idx = int(row["label_index"])
                pid = row.get("patient_id", "")
                self.entries.append((rel_path, label_idx, pid))

        if max_samples is not None and max_samples > 0:
            self.entries = self.entries[:max_samples]

        self.num_samples = len(self.entries)
        self.indices = np.arange(self.num_samples)
        self.rng = np.random.default_rng(self.seed)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        """Return total number of batches in one complete epoch."""
        if self.num_samples == 0:
            return 0
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self) -> None:
        """Permute sample indices at epoch boundary if shuffle is enabled."""
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, index: int) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load and return batch tensors for the specified batch index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Batch index {index} out of bounds for generator with {len(self)} batches.")

        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        current_batch_size = len(batch_indices)

        batch_x = np.empty((current_batch_size, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        batch_y = np.empty((current_batch_size,), dtype=np.int32)

        def _load_single(item: Tuple[int, int]) -> Tuple[int, np.ndarray, int]:
            i, idx = item
            rel_path, label_idx, _ = self.entries[idx]
            full_path = self.dicom_root / rel_path
            img_uint8 = load_dicom_as_array(full_path, target_size=self.target_size)
            img_norm = (img_uint8.astype(np.float32) / 127.5) - 1.0
            return i, img_norm, label_idx

        if self.num_workers > 1 and current_batch_size > 2:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.num_workers, current_batch_size)) as executor:
                for i, img_norm, label_idx in executor.map(_load_single, enumerate(batch_indices)):
                    batch_x[i] = img_norm
                    batch_y[i] = label_idx
        else:
            for i, idx in enumerate(batch_indices):
                _, img_norm, label_idx = _load_single((i, idx))
                batch_x[i] = img_norm
                batch_y[i] = label_idx

        if self.class_weights is not None:
            batch_sw = np.array([self.class_weights.get(int(lbl), 1.0) for lbl in batch_y], dtype=np.float32)
            return batch_x, batch_y, batch_sw

        return batch_x, batch_y

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Continuous generator yielding (batch_x, batch_y) indefinitely."""
        while True:
            for idx in range(len(self)):
                item = self[idx]
                yield item[0], item[1]
            self.on_epoch_end()

    @property
    def total_samples(self) -> int:
        return self.num_samples

    @property
    def steps_per_epoch(self) -> int:
        return len(self)

    @property
    def class_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for _, lbl, _ in self.entries:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    @property
    def unique_patients(self) -> Set[str]:
        return {pid for _, _, pid in self.entries if pid}

    def compute_balanced_class_weights(self) -> Dict[int, float]:
        """Compute balanced class weights exclusively from this manifest's distribution.
        
        Formula: w_j = N / (K * N_j) where K is number of classes (2).
        """
        counts = self.class_counts
        n_total = self.num_samples
        n_classes = len(counts)
        weights = {}
        for c, count in counts.items():
            if count > 0:
                weights[c] = float(n_total / (n_classes * count))
            else:
                weights[c] = 1.0
        return weights


def rsna_batch_generator(
    manifest_csv: Path | str,
    dicom_root: Path | str,
    batch_size: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    seed: int = RANDOM_SEED
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Memory-efficient streaming batch generator reading DICOMs on-the-fly from disk.

    Yields:
        tuple of (batch_images, batch_labels) where:
            batch_images: shape (batch_size, 224, 224, 3) float32 in range [-1.0, 1.0] (MobileNetV2 format)
            batch_labels: shape (batch_size,) int32
    """
    gen = RSNABatchGenerator(
        manifest_csv=manifest_csv,
        dicom_root=dicom_root,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=shuffle,
        seed=seed,
    )
    return iter(gen)


def validate_local_dicoms(
    dicom_paths: List[Path],
    max_checks: Optional[int] = None,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """Validate readable DICOM headers and pixel arrays for a list of DICOM paths.

    Memory-safe: checks files individually and releases pixel arrays immediately.
    High-throughput: uses conservative worker thread pool for parallel I/O.

    Returns:
        dictionary with keys: checked_count, valid_count, corrupt_count, corrupt_files, unchecked_count
    """
    if pydicom is None:
        raise ImportError("pydicom is required to validate DICOM images.")

    to_check = dicom_paths if max_checks is None else dicom_paths[:max_checks]
    if not to_check:
        return {
            "checked_count": 0,
            "valid_count": 0,
            "corrupt_count": 0,
            "corrupt_files": [],
            "unchecked_count": len(dicom_paths),
        }

    def _validate_single_dicom(p: Path) -> Tuple[str, bool, Optional[str]]:
        try:
            if not p.is_file() or p.stat().st_size == 0:
                return str(p), False, "File does not exist or has zero size"
            ds = pydicom.dcmread(str(p), stop_before_pixels=False)
            arr = ds.pixel_array
            if arr is None or arr.size == 0:
                return str(p), False, "Pixel array is empty"
            return str(p), True, None
        except Exception as e:
            return str(p), False, str(e)

    valid_count = 0
    corrupt_files: List[Tuple[str, str]] = []

    if num_workers > 1 and len(to_check) > 16:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for p_str, is_valid, err_msg in executor.map(_validate_single_dicom, to_check):
                if is_valid:
                    valid_count += 1
                else:
                    corrupt_files.append((p_str, err_msg or "Unknown read error"))
    else:
        for p in to_check:
            p_str, is_valid, err_msg = _validate_single_dicom(p)
            if is_valid:
                valid_count += 1
            else:
                corrupt_files.append((p_str, err_msg or "Unknown read error"))

    return {
        "checked_count": len(to_check),
        "valid_count": valid_count,
        "corrupt_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "unchecked_count": len(dicom_paths) - len(to_check),
    }


def audit_rsna_pipeline(
    metadata_dir: Path | str,
    dicom_dir: Path | str,
    processed_dir: Path | str,
    max_readability_checks: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute an end-to-end dataset audit of the RSNA pipeline with distinct tier statuses."""
    meta_p = Path(metadata_dir)
    dicom_p = Path(dicom_dir)
    proc_p = Path(processed_dir)

    issues: List[str] = []
    warnings: List[str] = []

    # --------------------------------------------------------------------------
    # Tier 1: Metadata & Annotation Integrity
    # --------------------------------------------------------------------------
    mapping_file = meta_p / "pneumonia-challenge-dataset-mappings_2018.json"
    annotations_file = meta_p / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"

    metadata_status = "PASS"
    mappings: List[Dict[str, Any]] = []
    annotations: Dict[str, Any] = {}
    valid_records: List[RSNARecord] = []
    excluded: List[RSNAExcludedRecord] = []

    if not mapping_file.is_file():
        metadata_status = "FAIL"
        issues.append(f"Missing mappings JSON at {mapping_file}")

    if not annotations_file.is_file():
        metadata_status = "FAIL"
        issues.append(f"Missing annotations JSON at {annotations_file}")

    if metadata_status == "PASS":
        mappings, map_err = load_rsna_mapping(mapping_file)
        annotations, ann_err = load_rsna_annotations(annotations_file)
        if map_err:
            warnings.append(f"Mapping parser flagged {len(map_err)} entries with issues")
        valid_records, excluded = build_cohort_records(mappings, annotations, dicom_dir=dicom_p)

    # --------------------------------------------------------------------------
    # Tier 2: Manifest & Patient-Split Integrity
    # --------------------------------------------------------------------------
    manifest_status = "PASS"
    split_summary_p = proc_p / "split_summary.json"
    manifest_summary: Optional[Dict[str, Any]] = None
    patient_disjoint = False

    if split_summary_p.is_file():
        try:
            with open(split_summary_p, "r", encoding="utf-8") as f:
                manifest_summary = json.load(f)
            patient_disjoint = manifest_summary.get("patient_leakage_verified_zero", False)
            if not patient_disjoint:
                manifest_status = "FAIL"
                issues.append("Patient leakage detected across generated manifests!")
        except Exception as e:
            manifest_status = "FAIL"
            issues.append(f"Failed to read split summary JSON: {e}")
    else:
        manifest_status = "NOT_GENERATED"
        warnings.append(f"Processed manifests not found at {split_summary_p}. Run manifest generation.")

    # --------------------------------------------------------------------------
    # Tier 3: Local DICOM Completeness
    # --------------------------------------------------------------------------
    dicom_index = build_dicom_index(dicom_p) if dicom_p.is_dir() else {}
    local_count = len(dicom_index)
    expected_valid_cohort = len(valid_records) if valid_records else 29684

    if local_count >= expected_valid_cohort:
        completeness_status = "COMPLETE"
    elif local_count > 0:
        completeness_status = f"PARTIAL ({local_count} / {expected_valid_cohort})"
        warnings.append(
            f"Partial local DICOM availability: {local_count} present out of {expected_valid_cohort} expected. Full archive download pending."
        )
    else:
        completeness_status = f"MISSING (0 / {expected_valid_cohort})"
        warnings.append(f"No local DICOM files found under {dicom_p}.")

    # --------------------------------------------------------------------------
    # Tier 4: DICOM Readability Validation
    # --------------------------------------------------------------------------
    readability_status = "UNCHECKED"
    readability_results = {
        "checked_count": 0,
        "valid_count": 0,
        "corrupt_count": 0,
        "corrupt_files": [],
        "unchecked_count": local_count,
    }

    if local_count > 0 and pydicom is not None:
        local_paths = [dicom_p / rel_p for rel_p in dicom_index.values()]
        readability_results = validate_local_dicoms(local_paths, max_checks=max_readability_checks)
        if readability_results["corrupt_count"] > 0:
            readability_status = f"FAILED ({readability_results['corrupt_count']} corrupt files)"
            issues.append(f"Corrupt local DICOM files detected: {readability_results['corrupt_files'][:3]}")
        elif readability_results["checked_count"] == local_count:
            readability_status = f"VERIFIED ({readability_results['valid_count']} / {local_count} readable)"
        else:
            readability_status = f"PARTIALLY_VERIFIED ({readability_results['valid_count']} / {readability_results['checked_count']} checked, {readability_results['unchecked_count']} unchecked)"

    # --------------------------------------------------------------------------
    # Tier 5: Overall Training-Data Readiness
    # --------------------------------------------------------------------------
    is_ready_for_training = False
    if issues:
        overall_readiness = "NOT_READY (CRITICAL ISSUES)"
    elif completeness_status != "COMPLETE":
        overall_readiness = f"NOT_READY (PARTIAL LOCAL DICOMS: {local_count} / {expected_valid_cohort} present)"
    elif manifest_status != "PASS":
        overall_readiness = "NOT_READY (MANIFESTS INCOMPLETE)"
    elif "FAILED" in readability_status:
        overall_readiness = "NOT_READY (CORRUPT DICOMS DETECTED)"
    else:
        overall_readiness = "READY"
        is_ready_for_training = True

    # Assemble comprehensive report
    return {
        "status": "PASS" if not issues else "FAIL",
        "metadata_status": metadata_status,
        "manifest_status": manifest_status,
        "completeness_status": completeness_status,
        "readability_status": readability_status,
        "overall_readiness": overall_readiness,
        "is_ready_for_training": is_ready_for_training,
        "mapping_count": len(mappings),
        "total_valid_cohort": len(valid_records),
        "total_excluded_cohort": len(excluded),
        "unique_patients_in_valid_cohort": len({r.patient_id for r in valid_records}),
        "class_distribution": {
            LABEL_NO_LUNG_OPACITY: sum(1 for r in valid_records if r.label_name == LABEL_NO_LUNG_OPACITY),
            LABEL_LUNG_OPACITY: sum(1 for r in valid_records if r.label_name == LABEL_LUNG_OPACITY),
        } if valid_records else {},
        "local_dicoms_present_count": local_count,
        "readability_validation": readability_results,
        "manifest_summary": manifest_summary,
        "patient_leakage_verified_zero": patient_disjoint,
        "issues": issues,
        "warnings": warnings,
    }

