# Model Checkpoints & Serialization

This directory stores trained model checkpoints, architecture exports, and weights serialization files.

> [!CAUTION]
> **Git Ignore Policy**: Binary weights (`*.keras`, `*.h5`, `*.tflite`, `*.pt`) are excluded from version control via `.gitignore`. Store production model artifacts in external model registries or designated cloud buckets (e.g., S3, Hugging Face Hub, MLflow).

---

## 1. Expected Baseline Architecture

- **Backbone**: MobileNetV2 (pre-trained on ImageNet or trained from scratch).
- **Classification Head**:
  - `GlobalAveragePooling2D()`
  - `Dropout(rate=0.2)`
  - `Dense(units=1, activation='sigmoid')` (Probability of Pneumonia)
- **Loss Function**: `binary_crossentropy`
- **Primary Optimization**: Adam ($\text{learning rate} = 10^{-4}$)

---

## 2. Checkpoint Naming Standard

When training via `src/train.py`, checkpoints are saved using the convention:
- **Best Validation Metric Checkpoint**: `mobilenetv2_pneumonia_best.keras`
- **Final Epoch Model**: `mobilenetv2_pneumonia_final.keras`

---

## 3. Loading a Model for Inference

```python
import tensorflow as tf
from src.config import DEFAULT_MODEL_PATH

model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
```

