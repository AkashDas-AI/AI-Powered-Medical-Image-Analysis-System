"""Inference module for pediatric pneumonia chest X-ray classification.

Provides the single-image and batch prediction interface with contract validation,
preprocessed tensor alignment, and explicit clinical non-diagnostic disclaimers.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from PIL import Image

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    POSITIVE_CLASS,
    RSNA_BEST_MODEL_PATH,
    RSNA_CLASS_NAMES,
    RSNA_POSITIVE_CLASS,
)
from src.preprocessing import preprocess_for_model

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    TENSORFLOW_AVAILABLE = False


def load_inference_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """Load a trained Keras model for inference.

    Args:
        model_path: Path to serialized model (.keras or .h5).
            Defaults to best available RSNA or baseline checkpoint if None.

    Returns:
        Loaded Keras model instance.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If the model file does not exist.
    """
    if model_path is not None:
        resolved_path = Path(model_path)
    elif RSNA_BEST_MODEL_PATH.exists():
        resolved_path = RSNA_BEST_MODEL_PATH
    elif BEST_MODEL_PATH.exists():
        resolved_path = BEST_MODEL_PATH
    elif DEFAULT_MODEL_PATH.exists():
        resolved_path = DEFAULT_MODEL_PATH
    else:
        resolved_path = RSNA_BEST_MODEL_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {resolved_path}. "
            "Please train a model using 'python main.py train-rsna' or provide a valid --model-path."
        )

    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError(
            "TensorFlow is required to load models. Install via pip install -r requirements.txt"
        )

    try:
        model = tf.keras.models.load_model(resolved_path, compile=False)
    except Exception as exc:
        raise ValueError(
            f"Failed to load model checkpoint at {resolved_path.name}: {type(exc).__name__}. "
            "Please ensure the file is a valid, uncorrupted Keras model."
        ) from exc

    if not hasattr(model, "output_shape") or model.output_shape is None:
        raise ValueError(f"Incompatible model architecture loaded from {resolved_path.name}: missing output shape.")

    # Store resolved path on model instance for auditability
    setattr(model, "_checkpoint_path", str(resolved_path))
    return model


def predict_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[Union[Any, Callable[[np.ndarray], Any]]] = None,
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Execute forward inference for a single chest radiograph.

    Input Contract:
    - Ingests image path, PIL Image, or NumPy array.
    - Applies MobileNetV2 preprocessing (RGB conversion, 224x224 resize, [-1, 1] scaling).
    - Accepts pre-loaded model or loads from disk on demand.

    Output Contract:
    - predicted_class: 'NORMAL' or 'PNEUMONIA'
    - confidence: float in [0.5, 1.0] representing model confidence in the chosen class
    - raw_score: float in [0.0, 1.0] representing raw model activation for pneumonia
    - probabilities: dictionary with class-level scores
    - threshold: decision threshold applied
    - disclaimer: explicit clinical disclaimer highlighting non-calibrated scores

    Args:
        image_input: File path, PIL Image, or NumPy array.
        model: Optional pre-loaded Keras model or callable function.
        model_path: Path to model checkpoint if model is not pre-loaded.
        threshold: Decision threshold for the positive class (PNEUMONIA). Default: 0.5.

    Returns:
        Dict[str, Any]: Structured prediction response meeting the output contract.

    Raises:
        ValueError: If neither model nor existing model_path is provided.
        TypeError: If input types are invalid.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Decision threshold must be between 0.0 and 1.0, got {threshold}")

    # 1. Resolve model instance
    active_model = model
    if active_model is None:
        active_model = load_inference_model(model_path)

    # 2. Preprocess image into batch tensor (1, 224, 224, 3)
    tensor = preprocess_for_model(image_input)

    # 3. Model forward pass
    if callable(active_model):
        raw_output = active_model(tensor)
    elif hasattr(active_model, "predict"):
        raw_output = active_model.predict(tensor, verbose=0)
    else:
        raise TypeError(f"Provided model object of type {type(active_model)} is neither callable nor a Keras Model.")

    # Convert TensorFlow tensor to NumPy array if needed
    if hasattr(raw_output, "numpy"):
        pred_array = raw_output.numpy()
    else:
        pred_array = np.asarray(raw_output)

    # 4. Extract scalar pneumonia probability
    if pred_array.ndim == 2 and pred_array.shape[1] == 1:
        # Binary sigmoid output: shape (1, 1)
        prob_pneumonia = float(pred_array[0, 0])
    elif pred_array.ndim == 2 and pred_array.shape[1] == len(CLASS_NAMES):
        # Softmax output: shape (1, 2) where index 1 is PNEUMONIA
        prob_pneumonia = float(pred_array[0, 1])
    elif pred_array.ndim == 1 and pred_array.shape[0] == 1:
        prob_pneumonia = float(pred_array[0])
    else:
        # Fallback for scalar output
        prob_pneumonia = float(np.squeeze(pred_array))

    prob_pneumonia = float(np.clip(prob_pneumonia, 0.0, 1.0))
    prob_normal = float(np.clip(1.0 - prob_pneumonia, 0.0, 1.0))

    # 5. Determine class label and confidence
    if prob_pneumonia >= threshold:
        predicted_class = POSITIVE_CLASS
        confidence = prob_pneumonia
    else:
        predicted_class = "NORMAL"
        confidence = prob_normal

    return {
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "raw_score": round(float(prob_pneumonia), 4),
        "threshold": float(threshold),
        "probabilities": {
            "NORMAL": round(float(prob_normal), 4),
            "PNEUMONIA": round(float(prob_pneumonia), 4),
        },
        "disclaimer": (
            "NON-DIAGNOSTIC NOTICE: Model scores reflect exploratory neural network activations, "
            "NOT calibrated clinical probabilities. This software is not an FDA/CE-approved "
            "diagnostic medical device and must never substitute clinical judgment."
        ),
    }


def predict_rsna_dicom(
    dicom_input: Union[str, Path],
    model: Optional[Union[Any, Callable[[np.ndarray], Any]]] = None,
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    include_patient_id: bool = False,
) -> Dict[str, Any]:
    """Execute forward inference on a single RSNA chest DICOM image.

    Input Contract:
    - Ingests DICOM file path (str or Path).
    - Decodes pixel data, handles PhotometricInterpretation (MONOCHROME1 inversion),
      applies min-max window normalization, and resizes to (224, 224, 3) RGB.
    - Scales intensities to MobileNetV2 range [-1.0, 1.0].
    - Evaluates via loaded RSNA model checkpoint.

    Privacy & Security Contract:
    - By default (include_patient_id=False), sensitive identifiers (PatientID,
      StudyInstanceUID, SOPInstanceUID) are excluded from the output.
    - Only non-sensitive technical metadata (ViewPosition, PhotometricInterpretation,
      dimensions, modality) is exposed by default.
    - PatientID is opt-in for authorized research and debugging only.

    Output Contract:
    - model_checkpoint: filesystem path of model used
    - dicom_path: filesystem path of evaluated DICOM
    - predicted_class: 'LUNG_OPACITY' or 'NO_LUNG_OPACITY'
    - lung_opacity_score: float in [0.0, 1.0] representing model activation
    - decision_threshold: decision threshold applied (default 0.50)
    - is_opacity_detected: boolean flag (True if score >= threshold)
    - probabilities: dictionary with class scores
    - dicom_metadata: extracted non-sensitive DICOM header attributes
    - input_tensor_shape: [1, 224, 224, 3]
    - disclaimer: explicit clinical disclaimer emphasizing non-diagnostic research nature

    Args:
        dicom_input: Path to DICOM file.
        model: Optional pre-loaded Keras model or callable.
        model_path: Path to model checkpoint if not preloaded.
        threshold: Decision threshold in [0.0, 1.0]. Defaults to 0.50.
        include_patient_id: Opt-in flag to include de-identified PatientID in output. Default: False.

    Returns:
        Dict[str, Any]: Structured prediction dictionary.

    Raises:
        ValueError: If threshold is not in [0.0, 1.0] or DICOM is corrupt.
        FileNotFoundError: If DICOM file or model path does not exist.
        TypeError: If dicom_input is not a valid path.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Decision threshold must be between 0.0 and 1.0, got {threshold}")

    if not isinstance(dicom_input, (str, Path)):
        raise TypeError(f"dicom_input must be a str or Path, got {type(dicom_input).__name__}")

    dcm_path = Path(dicom_input)
    if not dcm_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dcm_path}")
    if not dcm_path.is_file():
        raise ValueError(f"DICOM path is not a file: {dcm_path}")

    # 1. Resolve model instance
    active_model = model
    if active_model is None:
        active_model = load_inference_model(model_path)

    # 2. Extract non-sensitive DICOM header metadata (Privacy-by-Default)
    metadata: Dict[str, Any] = {}
    try:
        import pydicom
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        metadata = {
            "modality": str(getattr(ds, "Modality", "CR")),
            "view_position": str(getattr(ds, "ViewPosition", "UNKNOWN")),
            "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
        }
        if include_patient_id:
            metadata["patient_id"] = str(getattr(ds, "PatientID", "UNKNOWN"))
    except Exception as exc:
        metadata = {"error": f"Could not read DICOM header: {type(exc).__name__}"}

    # 3. Decode DICOM and preprocess for MobileNetV2 using unified shared pipeline
    from src.preprocessing import preprocess_for_model
    tensor = preprocess_for_model(dcm_path, target_size=(224, 224))

    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"Preprocessed tensor contains non-finite values for {dcm_path.name}")

    # 4. Model forward pass
    if callable(active_model):
        raw_output = active_model(tensor)
    elif hasattr(active_model, "predict"):
        raw_output = active_model.predict(tensor, verbose=0)
    else:
        raise TypeError(f"Provided model of type {type(active_model)} is neither callable nor a Keras Model.")

    if hasattr(raw_output, "numpy"):
        pred_array = raw_output.numpy()
    else:
        pred_array = np.asarray(raw_output)

    prob_opacity = float(np.squeeze(pred_array))
    prob_opacity = float(np.clip(prob_opacity, 0.0, 1.0))
    prob_no_opacity = float(np.clip(1.0 - prob_opacity, 0.0, 1.0))

    predicted_class = RSNA_POSITIVE_CLASS if prob_opacity >= threshold else "NO_LUNG_OPACITY"
    checkpoint_name = getattr(active_model, "_checkpoint_path", str(model_path or "in-memory-model"))

    return {
        "model_checkpoint": checkpoint_name,
        "dicom_path": str(dcm_path),
        "predicted_class": predicted_class,
        "lung_opacity_score": round(prob_opacity, 4),
        "decision_threshold": float(threshold),
        "is_opacity_detected": bool(prob_opacity >= threshold),
        "probabilities": {
            "NO_LUNG_OPACITY": round(prob_no_opacity, 4),
            "LUNG_OPACITY": round(prob_opacity, 4),
        },
        "dicom_metadata": metadata,
        "input_tensor_shape": list(tensor.shape),
        "disclaimer": (
            "EDUCATIONAL RESEARCH PROTOTYPE ONLY: This model is developed for research and "
            "educational benchmarking on chest radiographic opacity classification. It is not "
            "an FDA-cleared medical device, has not been validated in clinical prospective trials, "
            "and MUST NOT be used for diagnostic, prognostic, or treatment decisions. Model scores "
            "represent uncalibrated neural network feature activations, not confirmed clinical diagnoses."
        ),
    }


