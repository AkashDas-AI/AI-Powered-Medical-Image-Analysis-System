# Dataset Specifications & Data Governance

This document records the datasets ingested, their clinical context, preprocessing pipelines, class distributions, and data splitting criteria.

---

## 1. Supported Modalities & Dataset Registry

| Dataset Identifier | Clinical Domain | Modality | Target Task | Dimensionality | Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *To be specified upon selection* | Thoracic / Pulmonary | Chest X-Ray / CT | Classification / Lesion Detection | 2D / 3D | DICOM / PNG |
| *Alternative benchmarks* | Brain MRI / Histopathology | MRI / WSI | Segmentation / Anomaly Detection | 3D / 2D Gigapixel | NIfTI / SVS |

*(This table will be populated dynamically as specific datasets such as NIH ChestX-ray14, CheXpert, BraTS, or custom clinical cohorts are integrated).*

---

## 2. Partitioning Strategy & Patient Independence Caveat

To prevent data leakage across patient exams, all newly constructed clinical splits should enforce **Patient-Level Splitting**:

- **Train Set (70% - 80%)**: Used solely for parameter optimization via backpropagation.
- **Validation Set (10% - 15%)**: Used for hyperparameter tuning, checkpointing, and early stopping.
- **Test Set (10% - 15%)**: Held out completely until final model benchmarking.

> [!WARNING]
> **Dataset Specific Caveat (Kermany et al. Kaggle Release)**:
> In the canonical public Kaggle release of the pediatric chest X-ray dataset, patient-level metadata is absent for the `NORMAL` class (files named `IM-xxxx-xxxx.jpeg`). Therefore, **patient-level independence across the original train/val/test splits cannot be verified**, and claims of guaranteed patient separation must not be asserted without external metadata linkage.

---

## 3. Data Preprocessing & Quality Assurance Protocol

1. **De-identification**: Removal of Protected Health Information (PHI) from DICOM headers in compliance with HIPAA / GDPR.
2. **Quality Control Check**:
   - Detection of corrupt files, zero-variance pixel matrices, or missing slices.
   - Dynamic dynamic range validation (validating HU window values or bit depths).
3. **Class Balance Profiling**:
   - Automated frequency extraction for each diagnostic class.
   - Calculation of positive-to-negative sample ratios for loss weighting.

