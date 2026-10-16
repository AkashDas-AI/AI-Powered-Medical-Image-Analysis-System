"""Unit tests for the model prediction interface and contract verification.

Verifies the output dictionary contract, confidence score calculations, custom
threshold boundary handling, and defensive error handling without requiring
full dataset downloads or trained checkpoint files.
"""

from typing import Any, Dict

import numpy as np
import pytest
from PIL import Image

from src.predict import predict_image


@pytest.fixture
def dummy_radiograph() -> Image.Image:
    """Create a synthetic 224x224 grayscale radiograph image."""
    arr = np.random.randint(50, 200, size=(224, 224), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def test_predict_contract_positive_pneumonia(dummy_radiograph: Image.Image) -> None:
    """Verify prediction contract when model outputs high pneumonia probability."""
    # Mock model callable returning 0.88 pneumonia probability
    mock_model = lambda tensor: np.array([[0.88]], dtype=np.float32)

    result = predict_image(dummy_radiograph, model=mock_model, threshold=0.5)

    # 1. Structural contract validation
    assert isinstance(result, dict)
    for expected_key in (
        "predicted_class",
        "confidence",
        "raw_score",
        "threshold",
        "probabilities",
        "disclaimer",
    ):
        assert expected_key in result, f"Missing key in prediction contract: {expected_key}"

    # 2. Value validation
    assert result["predicted_class"] == "PNEUMONIA"
    assert result["confidence"] == 0.88
    assert result["raw_score"] == 0.88
    assert result["threshold"] == 0.5
    assert result["probabilities"]["PNEUMONIA"] == 0.88
    assert np.isclose(result["probabilities"]["NORMAL"], 0.12, atol=1e-4)

    # 3. Medical disclaimer check
    assert "NON-DIAGNOSTIC NOTICE" in result["disclaimer"]


def test_predict_contract_normal(dummy_radiograph: Image.Image) -> None:
    """Verify prediction contract when model outputs low pneumonia probability (NORMAL)."""
    # Mock model callable returning 0.12 pneumonia probability
    mock_model = lambda tensor: np.array([[0.12]], dtype=np.float32)

    result = predict_image(dummy_radiograph, model=mock_model, threshold=0.5)

    assert result["predicted_class"] == "NORMAL"
    assert result["confidence"] == 0.88  # 1.0 - 0.12
    assert result["raw_score"] == 0.12
    assert result["probabilities"]["NORMAL"] == 0.88
    assert result["probabilities"]["PNEUMONIA"] == 0.12


def test_predict_custom_threshold(dummy_radiograph: Image.Image) -> None:
    """Verify custom decision thresholding modifies class assignment appropriately."""
    # Score 0.65 is positive at default 0.5, but negative if threshold is set to 0.70
    mock_model = lambda tensor: np.array([[0.65]], dtype=np.float32)

    # At threshold 0.5 -> PNEUMONIA
    result_default = predict_image(dummy_radiograph, model=mock_model, threshold=0.5)
    assert result_default["predicted_class"] == "PNEUMONIA"

    # At threshold 0.70 -> NORMAL
    result_conservative = predict_image(dummy_radiograph, model=mock_model, threshold=0.70)
    assert result_conservative["predicted_class"] == "NORMAL"
    assert result_conservative["threshold"] == 0.70


def test_predict_with_softmax_multi_output(dummy_radiograph: Image.Image) -> None:
    """Verify predict_image handles 2-unit softmax outputs [[prob_normal, prob_pneumonia]]."""
    # Mock model returning [[0.10, 0.90]]
    mock_model = lambda tensor: np.array([[0.10, 0.90]], dtype=np.float32)

    result = predict_image(dummy_radiograph, model=mock_model, threshold=0.5)
    assert result["predicted_class"] == "PNEUMONIA"
    assert result["confidence"] == 0.90
    assert result["probabilities"]["PNEUMONIA"] == 0.90


def test_predict_invalid_threshold_error(dummy_radiograph: Image.Image) -> None:
    """Verify ValueError is raised when threshold is out of [0.0, 1.0] bounds."""
    mock_model = lambda tensor: np.array([[0.5]], dtype=np.float32)

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        predict_image(dummy_radiograph, model=mock_model, threshold=1.5)

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        predict_image(dummy_radiograph, model=mock_model, threshold=-0.1)


def test_predict_missing_model_checkpoint(dummy_radiograph: Image.Image) -> None:
    """Verify FileNotFoundError is raised when pointing to a non-existent checkpoint path."""
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        predict_image(
            dummy_radiograph,
            model=None,
            model_path="non_existent_model_checkpoint_9999.keras",
        )

