"""Model definition module for MobileNetV2 pediatric pneumonia classification.

Constructs, configures, and compiles the baseline deep neural network with
transfer learning feature extraction and task-specific classification head.
"""

from typing import Any, List, Optional, Tuple, Union

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses, metrics
    TENSORFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    layers = None  # type: ignore
    models = None  # type: ignore
    optimizers = None  # type: ignore
    losses = None  # type: ignore
    metrics = None  # type: ignore
    TENSORFLOW_AVAILABLE = False

from src.config import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_LEARNING_RATE,
    INPUT_SHAPE,
)


def _check_tensorflow() -> None:
    """Validate that TensorFlow is installed before invoking model operations."""
    if not TENSORFLOW_AVAILABLE or tf is None:
        raise ImportError(
            "TensorFlow is required to build and train models, but it is not installed. "
            "Install dependencies with: pip install -r requirements.txt"
        )


def build_mobilenet_v2(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 1,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    freeze_base: bool = True,
    weights: Optional[str] = "imagenet",
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
) -> Any:  # Returns tf.keras.Model when tf is present
    """Construct and compile the MobileNetV2 pneumonia classifier.

    Architecture:
    1. Input layer: (224, 224, 3) float32 normalized image tensors.
    2. MobileNetV2 base (optionally frozen for transfer learning).
    3. Global Average Pooling 2D layer reducing spatial dimensions to a feature vector.
    4. Dropout regularization layer (default rate: 0.2).
    5. Output classification layer:
       - Single unit with Sigmoid activation for binary classification (default).
       - N units with Softmax activation for multi-class taxonomy.

    Args:
        input_shape: 3D tuple defining image shape (H, W, C). Defaults to (224, 224, 3).
        num_classes: Number of output classes. Defaults to 1 (binary sigmoid).
        learning_rate: Learning rate for the Adam optimizer. Defaults to 1e-4.
        freeze_base: If True, locks pre-trained backbone parameters during initial training.
        weights: Pre-trained weights to load ('imagenet' or None). Defaults to 'imagenet'.
        dropout_rate: Dropout probability before the classification head. Defaults to 0.2.

    Returns:
        tf.keras.Model: Compiled Keras model instance.

    Raises:
        ImportError: If TensorFlow is not available in the active environment.
        ValueError: If input_shape or num_classes parameters are invalid.
    """
    _check_tensorflow()

    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3D tuple (H, W, C), got {input_shape}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")

    # 1. Instantiate MobileNetV2 feature extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = not freeze_base

    # 2. Build top classification head directly from backbone output
    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # 3. Output layer and metric selection
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="lung_opacity_prob")(x)
        loss_fn = losses.BinaryCrossentropy()
        metric_list: List[Union[str, metrics.Metric]] = [
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
        loss_fn = losses.CategoricalCrossentropy()
        metric_list = ["accuracy"]

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_pneumonia_classifier")

    # 4. Compile model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metric_list)

    return model