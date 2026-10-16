"""Smoke tests for model building, layer connectivity, and Grad-CAM explainability.

Executes minimal model construction and gradient tracing using weights=None
to ensure zero external downloads, lightweight execution, and offline compatibility.
"""

import numpy as np
import pytest
from PIL import Image

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow is not installed")
def test_build_mobilenet_v2_architecture_contract() -> None:
    """Verify MobileNetV2 builds with expected input/output contract without downloading weights."""
    from src.model import build_mobilenet_v2

    model = build_mobilenet_v2(
        input_shape=(224, 224, 3),
        num_classes=1,
        weights=None,  # Offline unweighted initialization
        freeze_base=True,
    )

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)

    # Test forward pass on synthetic input tensor
    dummy_input = np.random.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (1, 1)
    assert 0.0 <= float(output.numpy()[0, 0]) <= 1.0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow is not installed")
def test_gradcam_and_overlay_smoke() -> None:
    """Verify Grad-CAM computes gradients and overlays without graph disconnection."""
    from src.explainability import find_target_conv_layer, generate_gradcam, overlay_heatmap
    from src.model import build_mobilenet_v2

    model = build_mobilenet_v2(weights=None)
    conv_layer = find_target_conv_layer(model)
    assert conv_layer == "Conv_1"

    dummy_input = np.random.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
    heatmap = generate_gradcam(model, dummy_input)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.shape == (7, 7)
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0)

    # Test visual overlay
    dummy_img = Image.new("RGB", (224, 224), color="gray")
    blended = overlay_heatmap(dummy_img, heatmap)
    assert isinstance(blended, Image.Image)
    assert blended.size == (224, 224)
    assert blended.mode == "RGB"

