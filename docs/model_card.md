# Model Card: MobileNetV2 Pediatric Pneumonia Classifier

## 1. Model Details
- **Model Architecture**: MobileNetV2 with Custom Binary Classification Head
- **Backbone Framework**: TensorFlow / Keras 2.x
- **Input Dimensions**: $(224, 224, 3)$ with float32 pixel values normalized to $[-1.0, 1.0]$
- **Output**: Single scalar value $\in [0.0, 1.0]$ representing the predicted probability of pneumonia
- **Decision Threshold**: Benchmark operational threshold fixed at $\tau = 0.50$. Speculative or uncalibrated threshold tuning is omitted without prospective clinical utility analysis.
- **Primary Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam ($\text{learning rate} = 10^{-4}$)

---

## 2. Intended Use & Applications
- **Primary Use**: Educational demonstration, academic research, and reference implementation for lightweight medical computer vision pipelines.
- **Intended Users**: Machine learning researchers, computer vision students, biomedical engineering developers.
- **Out-of-Scope / Prohibited Uses**:
  - Direct clinical diagnosis, screening, triage, or treatment planning in real-world healthcare settings.
  - Diagnostic evaluation of adult chest radiographs without rigorous domain-specific retraining and regulatory approval.
  - Standalone clinical decision-making without expert radiologist oversight.

---

## 3. Evaluation & Benchmarking Metrics

The model pipeline computes the following clinical screening metrics:
- **Sensitivity (Recall)**: $\frac{TP}{TP + FN}$ — Minimizing false negatives is critical in medical screening to avoid missing positive infections.
- **Specificity**: $\frac{TN}{TN + FP}$ — Quantifying false alarm rates.
- **AUROC (Area Under the ROC Curve)**: Threshold-independent assessment of discriminative capability.
- **F1-Score**: Harmonic mean of Precision and Recall.

---

## 4. Calibration & Interpretability Caveats

> [!CAUTION]
> **Probability Calibration**: The model's raw output probabilities (sigmoid scores) reflect network confidence, NOT calibrated clinical risk probabilities. They must not be interpreted as absolute statistical likelihoods of disease presence.

> [!NOTE]
> **Exploratory Nature of Grad-CAM**: Visual saliency heatmaps generated via Grad-CAM highlight spatial regions that strongly influenced the model's intermediate representations in the `Conv_1` layer ($7 \times 7 \times 1280$). They do NOT constitute radiological proof of lesion boundaries, bacterial consolidation, or viral infiltrates.

---

## 5. RSNA 2018 Adjudicated Baseline Benchmarks

- **Target Checkpoint**: `models/mobilenetv2_rsna_best.keras` (9,636,581 bytes)
- **Dataset**: RSNA Pneumonia Detection Challenge 2018 (29,684 adjudicated DICOM cases, patient-level leak-free split)
- **Target Label**: Visual radiographic lung opacity (expert radiologist consensus annotation), NOT confirmed clinical pneumonia or microbiological infection.
- **Training Coverage**: 1 complete epoch across the full training cohort (20,913 images / 1,308 batches) and validation cohort (4,370 images / 274 batches) with bounded streaming generator and inverse class weights. Multi-epoch convergence is not established; checkpoint serves as an audited transfer-learning baseline.
- **Privacy Controls**: Sensitive DICOM metadata (`PatientID`, `StudyInstanceUID`, `SOPInstanceUID`) are excluded by default from inference outputs, console displays, and JSON reports. Opt-in via `--include-patient-id`.
- **Evaluation Split**: Held-Out Test Set (4,401 DICOMs: 3,371 No Lung Opacity, 1,030 Lung Opacity)
- **Operational Decision Threshold**: $\tau = 0.50$ (fixed research benchmark)
- **Measured Metrics**:
  - **AUROC**: `0.8238`
  - **Accuracy**: `74.91%` (3,297 / 4,401)
  - **Sensitivity (Recall)**: `77.18%` (795 / 1,030 opacities detected)
  - **Specificity**: `74.22%` (2,502 / 3,371 non-opacities ruled out)
  - **Precision (PPV)**: `47.78%` (795 / 1,664 positive predictions)
  - **F1-Score**: `0.5902`
- **Confusion Matrix**: $TN = 2502$, $FP = 869$, $FN = 235$, $TP = 795$
- **CLI Inference Command**: `python main.py predict-rsna --image <path.dcm> --gradcam`


