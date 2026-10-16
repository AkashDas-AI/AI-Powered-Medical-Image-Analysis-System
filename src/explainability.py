"""Explainability module implementing Gradient-weighted Class Activation Mapping (Grad-CAM).

Generates exploratory visual saliency heatmaps highlighting convolutional feature
activations associated with the model's prediction.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    TENSORFLOW_AVAILABLE = False


def find_target_conv_layer(model: Any) -> str:
    """Locate the deepest 2D convolutional layer within the model architecture.

    Args:
        model: Keras model instance.

    Returns:
        str: Name of the identified convolutional layer.

    Raises:
        ValueError: If no 2D convolutional layer is found in the model.
    """
    # 1. Prioritize true Conv2D layer instances
    if TENSORFLOW_AVAILABLE and tf is not None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

    # 2. Name-based search excluding batch-norm and activation wrappers
    for layer in reversed(model.layers):
        name_lower = layer.name.lower()
        if "conv" in name_lower and not name_lower.endswith(("_bn", "_relu", "_act")):
            return layer.name

    # 3. Known default convolutional feature layer for MobileNetV2
    return "Conv_1"


def generate_gradcam(
    model: Any,
    image_tensor: np.ndarray,
    target_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """Compute a Grad-CAM saliency heatmap for an input image tensor.

    > [!IMPORTANT]
    > **Exploratory Caveat**: Grad-CAM heatmaps highlight receptive field features
    > that contributed to model classification. They do NOT delineate clinical lesion
    > margins, consolidations, or bacterial/viral pathology boundaries.

    Args:
        model: Compiled or loaded Keras model.
        image_tensor: Preprocessed 4D numpy array with shape (1, 224, 224, 3).
        target_layer_name: Optional specific convolutional layer name. If None,
            the deepest convolutional layer is automatically detected.
        class_index: Target class index. For binary models, defaults to 0 (Pneumonia probability).

    Returns:
        np.ndarray: 2D float32 array normalized to [0.0, 1.0] representing the heatmap.

    Raises:
        ImportError: If TensorFlow is not installed.
        ValueError: If input tensor dimensions are invalid.
    """
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError(
            "TensorFlow is required to generate Grad-CAM heatmaps. "
            "Install dependencies via pip install -r requirements.txt"
        )

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(f"image_tensor must have shape (1, H, W, C), got {image_tensor.shape}")

    # 1. Resolve target layer
    resolved_layer_name = target_layer_name or find_target_conv_layer(model)

    # 2. Build multi-output gradient model
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(resolved_layer_name).output, model.output],
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to access layer '{resolved_layer_name}' in model: {exc}"
        ) from exc

    # 3. Compute gradients with GradientTape
    img_tensor = tf.cast(image_tensor, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is not None and predictions.shape[-1] > 1:
            loss = predictions[:, class_index]
        else:
            loss = predictions[:, 0]

    # 4. Extract gradients of loss w.r.t. convolutional feature maps
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(
            f"Gradients could not be computed for target layer '{resolved_layer_name}'."
        )

    # 5. Global average pooling of gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Weight feature map channels by pooled gradients
    conv_outputs_2d = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs_2d), axis=-1)

    # 7. Apply ReLU and normalize to [0.0, 1.0]
    cam = tf.maximum(cam, 0.0)
    
    # Check for finite values
    if not tf.reduce_all(tf.math.is_finite(cam)):
        raise ValueError("Grad-CAM computation produced non-finite (NaN/Inf) values.")

    max_val = tf.math.reduce_max(cam)
    if max_val > 0.0:
        cam = cam / max_val

    return cam.numpy().astype(np.float32)


def overlay_heatmap(
    original_image: Union[str, Path, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> Image.Image:
    """Blend a 2D Grad-CAM heatmap onto the original chest radiograph.

    Args:
        original_image: Original image as PIL Image, file path (str or Path), or numpy array.
            Supports standard image formats (JPEG/PNG) and medical DICOM (.dcm) files.
        heatmap: 2D float array in [0.0, 1.0].
        alpha: Opacity factor for the heatmap overlay (0.0 to 1.0). Default: 0.4.
        colormap_name: Matplotlib colormap identifier. Default: 'jet'.

    Returns:
        PIL.Image.Image: Blended RGB visualization image.
    """
    if isinstance(original_image, (str, Path)):
        file_path = Path(original_image)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        if file_path.suffix.lower() == ".dcm":
            from src.rsna_dataset import load_dicom_as_array
            arr_uint8 = load_dicom_as_array(file_path)
            pil_base = Image.fromarray(arr_uint8).convert("RGB")
        else:
            pil_base = Image.open(file_path).convert("RGB")
    elif isinstance(original_image, Image.Image):
        pil_base = original_image.convert("RGB")
    elif isinstance(original_image, np.ndarray):
        arr = original_image
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.min() < 0.0:  # Normalized [-1.0, 1.0]
                arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            elif arr.max() <= 1.0:  # Scaled [0.0, 1.0]
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            pil_base = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            pil_base = Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
        else:
            pil_base = Image.fromarray(arr).convert("RGB")
    else:
        raise TypeError(f"Unsupported original_image type: {type(original_image)}")

    # If heatmap is entirely zero / degenerate, return base image directly to avoid misleading decorative color wash
    if heatmap.size == 0 or np.all(heatmap <= 0.0):
        return pil_base

    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray((heatmap * 255.0).astype(np.uint8), mode="L")
    resample_filter = getattr(Image, "Resampling", Image).BILINEAR
    heatmap_resized = heatmap_pil.resize(pil_base.size, resample=resample_filter)
    heatmap_arr = np.asarray(heatmap_resized, dtype=np.float32) / 255.0

    # Colorize heatmap
    try:
        import matplotlib as mpl
        import matplotlib.cm as cm
        if hasattr(mpl, "colormaps"):
            cmap = mpl.colormaps[colormap_name]
        else:
            cmap = cm.get_cmap(colormap_name)
        colored_cam = cmap(heatmap_arr)[:, :, :3]  # Discard alpha, shape (H, W, 3) in [0, 1]
        colored_cam_uint8 = (colored_cam * 255.0).astype(np.uint8)
        colored_pil = Image.fromarray(colored_cam_uint8, mode="RGB")
    except (ImportError, AttributeError, KeyError):  # Fallback: simple red/yellow overlay if matplotlib is absent
        red_channel = (heatmap_arr * 255.0).astype(np.uint8)
        green_channel = ((1.0 - heatmap_arr) * 128.0).astype(np.uint8)
        blue_channel = np.zeros_like(red_channel)
        colored_cam_uint8 = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        colored_pil = Image.fromarray(colored_cam_uint8, mode="RGB")

    # Blend original and colored heatmap
    blended = Image.blend(pil_base, colored_pil, alpha=alpha)
    return blended


def generate_gradcam_figure(
    original_image: Union[str, Path, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    predicted_class: str = "LUNG_OPACITY",
    model_score: float = 0.5,
    threshold: float = 0.5,
    target_layer_name: str = "Conv_1",
    alpha: float = 0.4,
    colormap_name: str = "jet",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Create a structured 3-panel Grad-CAM visualization with colorbar and clinical disclaimer.

    Panels:
    1. Original preprocessed chest radiograph
    2. Convolutional feature saliency map (with colorbar legend)
    3. Blended saliency overlay (CXR + Heatmap)

    Args:
        original_image: Input image (path, PIL Image, or array).
        heatmap: 2D numpy array with normalized saliency values in [0.0, 1.0].
        predicted_class: Model classification label ('LUNG_OPACITY' or 'NO_LUNG_OPACITY').
        model_score: Raw sigmoid activation score from the model.
        threshold: Decision threshold used for classification.
        target_layer_name: Name of the convolutional layer analyzed.
        alpha: Overlay opacity blending factor.
        colormap_name: Matplotlib colormap name.
        save_path: Optional file path to save the generated figure.

    Returns:
        matplotlib.figure.Figure: The rendered figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Resolve base image to uint8 RGB array
    if isinstance(original_image, (str, Path)):
        file_path = Path(original_image)
        if file_path.suffix.lower() == ".dcm":
            from src.rsna_dataset import load_dicom_as_array
            base_arr = load_dicom_as_array(file_path, target_size=(224, 224))
        else:
            pil = Image.open(file_path).convert("RGB")
            base_arr = np.array(pil.resize((224, 224), getattr(Image, "Resampling", Image).BILINEAR))
    elif isinstance(original_image, Image.Image):
        base_arr = np.array(original_image.convert("RGB").resize((224, 224), getattr(Image, "Resampling", Image).BILINEAR))
    elif isinstance(original_image, np.ndarray):
        arr = original_image
        if arr.ndim == 4:
            arr = arr[0]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.min() < 0.0:
                arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
            elif arr.max() <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        base_arr = arr
    else:
        raise TypeError(f"Unsupported original_image type: {type(original_image)}")

    is_degenerate = bool(heatmap.size == 0 or np.all(heatmap <= 0.0))
    title_suffix = " [Zero Positive Attribution / Degenerate]" if is_degenerate else ""

    # 2. Render 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"Exploratory Grad-CAM Saliency Analysis\n"
        f"Predicted: {predicted_class} (Model Score: {model_score:.4f} | "
        f"Threshold: {threshold:.2f} | Layer: {target_layer_name}){title_suffix}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Original CXR
    axes[0].imshow(base_arr)
    axes[0].set_title("1. Original Chest Radiograph", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Saliency Map with Colorbar Legend
    im_cam = axes[1].imshow(heatmap, cmap=colormap_name, vmin=0.0, vmax=1.0)
    p2_title = f"2. {target_layer_name} Saliency (No Positive Attribution)" if is_degenerate else f"2. {target_layer_name} Feature Saliency Map"
    axes[1].set_title(p2_title, fontsize=11, fontweight="bold")
    axes[1].axis("off")
    cbar = fig.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Relative Saliency (0.0 = Low, 1.0 = High)", fontsize=9)

    # Panel 3: Blended Overlay
    if is_degenerate:
        axes[2].imshow(base_arr)
        axes[2].set_title("3. Overlay (No Attribution Hotspots)", fontsize=11, fontweight="bold")
        axes[2].axis("off")
    else:
        heatmap_pil = Image.fromarray((heatmap * 255.0).astype(np.uint8), mode="L")
        resample_filter = getattr(Image, "Resampling", Image).BILINEAR
        heatmap_resized = np.array(
            heatmap_pil.resize((base_arr.shape[1], base_arr.shape[0]), resample_filter),
            dtype=np.float32
        ) / 255.0

        cmap = plt.get_cmap(colormap_name)
        colored_cam = (cmap(heatmap_resized)[:, :, :3] * 255.0).astype(np.uint8)
        blended_arr = (float(1.0 - alpha) * base_arr.astype(np.float32) + float(alpha) * colored_cam.astype(np.float32))
        blended_uint8 = np.clip(blended_arr, 0, 255).astype(np.uint8)

        axes[2].imshow(blended_uint8)
        axes[2].set_title(f"3. Blended Saliency Overlay (\u03b1={alpha:.2f})", fontsize=11, fontweight="bold")
        axes[2].axis("off")

    # Footer clinical research disclaimer
    fig.text(
        0.5, 0.02,
        "RESEARCH & EDUCATIONAL PROTOTYPE ONLY: Grad-CAM highlights convolutional feature activations.\n"
        "It does not delineate clinical lesion margins or prove diagnostic accuracy. Not for clinical use.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


