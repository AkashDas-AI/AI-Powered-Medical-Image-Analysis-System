"""Centralized configuration module for the AI-Powered Medical Image Analysis System.

Defines filesystem paths, model hyperparameters, image transformation dimensions,
class taxonomies, and reproducible random seeds.
"""

from pathlib import Path
from typing import Dict, Tuple

# ==============================================================================
# 1. Project Directory Layout
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Models and serialization
MODELS_DIR: Path = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "mobilenetv2_pneumonia.keras"
BEST_MODEL_PATH: Path = MODELS_DIR / "mobilenetv2_pneumonia_best.keras"

# Outputs and reporting
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
IMAGES_DIR: Path = PROJECT_ROOT / "images"

# RSNA Dataset layout
RSNA_RAW_DIR: Path = RAW_DATA_DIR / "rsna"
RSNA_METADATA_DIR: Path = RSNA_RAW_DIR / "metadata"
RSNA_DICOM_DIR: Path = RSNA_RAW_DIR / "dicom"
RSNA_PROCESSED_DIR: Path = PROCESSED_DATA_DIR / "rsna"
RSNA_MAPPINGS_FILE: Path = RSNA_METADATA_DIR / "pneumonia-challenge-dataset-mappings_2018.json"
RSNA_ANNOTATIONS_FILE: Path = RSNA_METADATA_DIR / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
RSNA_TRAIN_MANIFEST: Path = RSNA_PROCESSED_DIR / "train_manifest.csv"
RSNA_VAL_MANIFEST: Path = RSNA_PROCESSED_DIR / "val_manifest.csv"
RSNA_TEST_MANIFEST: Path = RSNA_PROCESSED_DIR / "test_manifest.csv"

# RSNA Model artifacts
RSNA_BEST_MODEL_PATH: Path = MODELS_DIR / "mobilenetv2_rsna_best.keras"
RSNA_FINAL_MODEL_PATH: Path = MODELS_DIR / "mobilenetv2_rsna_final.keras"
SMOKE_TEST_MODEL_PATH: Path = MODELS_DIR / "smoke_test_checkpoint.keras"

# Ensure runtime directories exist
for directory in (MODELS_DIR, REPORTS_DIR, PLOTS_DIR, PROCESSED_DATA_DIR, RSNA_PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Reproducibility & Randomization
# ==============================================================================
RANDOM_SEED: int = 42

# ==============================================================================
# 3. Image Dimensions & Input Specifications
# ==============================================================================
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
IMAGE_SIZE: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".jpeg",
    ".jpg",
    ".png",
    ".bmp",
    ".tiff",
    ".dcm",
)

# ==============================================================================
# 4. Classification Taxonomy
# ==============================================================================
# Generic / baseline binary taxonomy
CLASS_NAMES: Tuple[str, ...] = ("NORMAL", "PNEUMONIA")
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
POSITIVE_CLASS: str = "PNEUMONIA"
DEFAULT_THRESHOLD: float = 0.5

# RSNA 2018 specific taxonomy (annotation-grounded: not definitive pneumonia)
RSNA_CLASS_NAMES: Tuple[str, ...] = ("NO_LUNG_OPACITY", "LUNG_OPACITY")
RSNA_NUM_CLASSES: int = len(RSNA_CLASS_NAMES)
RSNA_CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(RSNA_CLASS_NAMES)}
RSNA_INDEX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(RSNA_CLASS_NAMES)}
RSNA_POSITIVE_CLASS: str = "LUNG_OPACITY"

# ==============================================================================
# 5. Training Hyperparameters
# ==============================================================================
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_EPOCHS: int = 20
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_DROPOUT_RATE: float = 0.2

