# Clinical, Algorithmic & Ethical Limitations

This document explicitly outlines the technical, statistical, and clinical limitations inherent in the **AI-Powered Medical Image Analysis System**.

---

## 1. Non-Diagnostic Educational Scope

> [!CAUTION]
> **PRIMARY CLINICAL DISCLAIMER**:
> This software is strictly intended for **educational, academic, and technical demonstration purposes**. It is NOT a certified medical software device, has NOT undergone clinical trials, and is NOT approved by regulatory bodies (such as the US FDA or European CE-MDR) for diagnostic usage. Under no circumstances should this system be used to inform real-world patient diagnosis, prognosis, triage, or clinical management.

---

## 2. Model Confidence vs. Clinical Probability

Deep neural networks trained with cross-entropy loss are frequently **uncalibrated**—they tend to produce overconfident predictions near 0.0 or 1.0 even when encountering out-of-distribution inputs or borderline clinical cases.

- **Non-Calibrated Output**: A model prediction score of $0.94$ indicates high network activation confidence, **not** a $94\%$ statistical probability that the patient has pneumonia.
- **Prevalence Shift**: In clinical deployment, pneumonia prevalence varies by season, geography, and patient age. Predictions made without calibrating for local prevalence can result in skewed positive predictive values (PPV).

---

## 3. Explainability (Grad-CAM) Limitations

While Class Activation Mapping (Grad-CAM) provides useful qualitative insight into convolutional feature activations:
1. **Exploratory, Not Diagnostic**: Saliency heatmaps indicate which receptive fields contributed most to the gradient backpropagation; they do **not** delineate actual pathological lesion margins or bacterial infiltrates.
2. **Resolution Loss**: MobileNetV2 downsamples input images from $224 \times 224$ to low-resolution feature maps ($7 \times 7$). Bilinear upsampling of this coarse grid produces diffuse blobs that lack fine anatomical precision.
3. **Confounding Focus**: Neural networks can attend to clinical artifacts (e.g., radiographic orientation markers "L"/"R", pediatric positioning aids, detector grid lines) rather than pulmonary parenchymal changes.

---

## 4. Cohort & Distributional Limitations

1. **Pediatric Restriction**: The baseline dataset consists entirely of children aged 1 to 5 years. Adult lungs exhibit different anatomical proportions, rib ossification, and common chronic conditions (e.g., smoking-related changes, fibrosis) which this model cannot interpret.
2. **AP vs. PA Projections**: The dataset predominantly uses Anterior-Posterior (AP) views. Applying the model to Posteroanterior (PA) or lateral projections can result in severe performance degradation.
3. **Etiology Ambiguity**: In the raw dataset, bacterial and viral pneumonia are frequently grouped under a unified "Pneumonia" label, preventing reliable microbiological discrimination.

