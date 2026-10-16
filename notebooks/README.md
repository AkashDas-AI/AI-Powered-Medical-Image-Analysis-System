# Exploratory & Experimental Notebooks

This directory is designated for interactive research, exploratory data analysis (EDA), and prototype validation.

---

## Planned Notebook Roadmap

1. **`01_data_exploration.ipynb`**:
   - **Objectives**:
     - Visual inspection of pediatric chest X-ray samples (NORMAL vs. PNEUMONIA).
     - Pixel intensity distribution analysis and bit-depth checks.
     - Class balance profiling across train, validation, and test splits.
     - Assessment of image artifacts, aspect ratios, and pediatric anatomical variations.

2. **`02_model_experimentation.ipynb`**:
   - **Objectives**:
     - Evaluation of MobileNetV2 baseline vs. alternative architectures (e.g., EfficientNet, ResNet50).
     - Learning rate scheduling and hyperparameter experimentation.
     - Exploratory Grad-CAM saliency map visualization on true-positive and false-positive cases.
     - Confidence threshold tuning for screening sensitivity prioritization.

---

## Execution Guidelines
- Run notebooks using the project's virtual environment (`.venv`).
- Ensure dependencies are installed via `pip install -r requirements.txt`.
- Notebooks should consume modules directly from `src/` (e.g., `from src.preprocessing import preprocess_image`).

