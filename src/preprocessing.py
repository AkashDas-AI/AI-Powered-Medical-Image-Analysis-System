"""Image preprocessing pipeline for the AI-Powered Medical Image Analysis System.

Provides consistent, deterministic image transformations for training, evaluation,
and single-image inference matching the MobileNetV2 specification.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

from src.config import IMAGE_CHANNELS, IMAGE_SIZE


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess a single medical image for MobileNetV2 inference or training.

    Performs:
    1. Input resolution and validation (path, PIL Image, or NumPy array).
    2. Guaranteed 3-channel RGB conversion (handling grayscale X-rays or RGBA).
    3. Bilinear resizing to target dimensions (default: 224 x 224).
    4. MobileNetV2 intensity scaling: (pixel / 127.5) - 1.0 -> range [-1.0, 1.0].
    5. Conversion to standard float32 numpy array.

    Args:
        image_input: File path (str or Path), PIL Image object, or numpy array.
        target_size: Tuple of (height, width). Defaults to (224, 224).

    Returns:
        np.ndarray: Preprocessed image array of shape (height, width, 3)
            and dtype float32 with values in [-1.0, 1.0].

    Raises:
        TypeError: If image_input is None or an unsupported type.
        FileNotFoundError: If image_input is a path that does not exist.
        ValueError: If image dimensions or array formats are invalid.
    """
    if image_input is None:
        raise TypeError("Input image cannot be None.")

    # 1. Resolve and open image if given a path
    if isinstance(image_input, (str, Path)):
        file_path = Path(image_input)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        # Check for DICOM file (.dcm or .dicom)
        if file_path.suffix.lower() in (".dcm", ".dicom"):
            from src.rsna_dataset import load_dicom_as_array
            arr_uint8 = load_dicom_as_array(file_path, target_size=target_size)
            return (arr_uint8.astype(np.float32) / 127.5) - 1.0

        try:
            pil_img = Image.open(file_path)
            pil_img.load()  # Verify file is not corrupt
        except Exception as exc:
            raise ValueError(f"Failed to read image from {file_path}: {exc}") from exc

    elif isinstance(image_input, Image.Image):
        pil_img = image_input

    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy array is empty.")
        if image_input.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D or 3D numpy array, got {image_input.ndim}D array with shape {image_input.shape}."
            )

        # Handle float arrays normalized in [0, 1] or [-1, 1]
        arr = image_input
        if np.issubdtype(arr.dtype, np.floating):
            if arr.min() < 0.0 or arr.max() <= 1.0:
                # Assuming [0, 1], scale to [0, 255]
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Handle 2D grayscale
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L")
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                pil_img = Image.fromarray(arr[:, :, 0], mode="L")
            elif arr.shape[2] == 3:
                pil_img = Image.fromarray(arr, mode="RGB")
            elif arr.shape[2] == 4:
                pil_img = Image.fromarray(arr, mode="RGBA")
            else:
                raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")
    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input).__name__}. "
            "Expected str, Path, PIL.Image.Image, or np.ndarray."
        )

    # 2. Ensure 3-channel RGB representation
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 3. Resize to target dimensions
    resample_filter = getattr(Image, "Resampling", Image).BILINEAR
    if pil_img.size != (target_size[1], target_size[0]):
        pil_img = pil_img.resize((target_size[1], target_size[0]), resample=resample_filter)

    # 4. Convert to float32 numpy array
    img_array = np.asarray(pil_img, dtype=np.float32)

    # 5. MobileNetV2 normalization: scale [0, 255] -> [-1.0, 1.0]
    # Formula matches tf.keras.applications.mobilenet_v2.preprocess_input
    normalized_array = (img_array / 127.5) - 1.0

    # Ensure shape (H, W, 3)
    if normalized_array.shape != (target_size[0], target_size[1], IMAGE_CHANNELS):
        raise ValueError(
            f"Unexpected preprocessed shape {normalized_array.shape}, "
            f"expected {(target_size[0], target_size[1], IMAGE_CHANNELS)}"
        )

    return normalized_array.astype(np.float32)


def preprocess_for_model(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess an image and add a batch dimension for model input.

    Args:
        image_input: File path, PIL Image, or numpy array.
        target_size: Target (height, width). Defaults to (224, 224).

    Returns:
        np.ndarray: Batch tensor of shape (1, height, width, 3), float32.
    """
    tensor_2d = preprocess_image(image_input, target_size=target_size)
    return np.expand_dims(tensor_2d, axis=0)


def deprocess_image(normalized_tensor: np.ndarray) -> np.ndarray:
    """Invert MobileNetV2 scaling ([-1.0, 1.0] -> [0, 255] uint8) for visualization.

    Args:
        normalized_tensor: Float array with values scaled to [-1.0, 1.0].

    Returns:
        np.ndarray: uint8 image array scaled to [0, 255].
    """
    if normalized_tensor.ndim == 4:
        normalized_tensor = normalized_tensor[0]
    arr = (normalized_tensor + 1.0) * 127.5
    return np.clip(arr, 0, 255).astype(np.uint8)

