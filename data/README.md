# Data Directory Guide

This directory houses the chest radiograph imaging data, metadata, and generated manifests for the **AI-Powered Medical Image Analysis System**.

> [!CAUTION]
> **Data Governance & Git Exclusion**: Never commit raw medical images or processed manifests to Git. Both `data/raw/*` and `data/processed/*` are explicitly excluded in `.gitignore`.

---

## 1. Approved Dataset: RSNA Pneumonia Detection Challenge (2018)

The system is configured for the **RSNA Pneumonia Detection Challenge (2018)** cohort, consisting of 30,000 frontal chest radiographs with expert radiologist consensus annotations.

### 1.1 Directory Layout
```
data/
├── raw/
│   └── rsna/
│       ├── metadata/
│       │   ├── pneumonia-challenge-annotations-adjudicated-kaggle_2018.json  (52,003,071 B)
│       │   ├── pneumonia-challenge-dataset-mappings_2018.json                (17,587,143 B)
│       │   ├── dataset_description.pdf                                       (54,441 B)
│       │   ├── terms_of_use.pdf                                              (198,030 B)
│       │   └── acknowledgements.pdf                                          (241,963 B)
│       └── dicom/
│           ├── <SOPInstanceUID>.dcm                                          (1024x1024 uint8)
│           └── ...
└── processed/
    └── rsna/
        ├── train_manifest.csv                                                (20,913 images, 8,563 patients)
        ├── val_manifest.csv                                                  (4,370 images, 1,835 patients)
        ├── test_manifest.csv                                                 (4,401 images, 1,835 patients)
        ├── excluded_manifest.csv                                             (316 images, 41 patients)
        └── split_summary.json                                                (Structured audit summary)
```

---

## 2. Official Download Sources & Provenance

All files are hosted publicly by the Radiological Society of North America (RSNA) on Amazon S3:

| Component | Official S3 Endpoint | Uncompressed Size |
| :--- | :--- | :--- |
| **DICOM Image Archive** | `https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip` | 3.71 GB (30,000 `.dcm` files) |
| **Adjudicated Annotations** | `https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` | 49.59 MB (83,809 annotations) |
| **NIH-to-RSNA Mapping** | `https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json` | 16.77 MB (30,000 mappings) |
| **Terms of Use** | `https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf` | 198 KB |

---

## 3. Label Taxonomy & Mapping

The binary screening task uses the following standardized classes:
- **`LUNG_OPACITY` (Class 1, Positive)**: Adjudicated consensus `Lung Opacity` ($n = 7,106$).
- **`NO_LUNG_OPACITY` (Class 0, Negative)**: Adjudicated consensus `Normal` ($n = 9,790$) and `No Lung Opacity / Not Normal` ($n = 12,788$).
- **Excluded Cases**: Radiologist-flagged exclusions ($n = 120$) and unresolved non-consensus studies ($n = 196$) are recorded in `excluded_manifest.csv`.

---

## 4. Pipeline Execution Commands

### 4.1 Audit the RSNA Pipeline
```bash
python main.py audit-rsna
```
Inspects metadata integrity, counts local DICOM files, verifies split summary metrics, and confirms zero cross-split patient leakage.

### 4.2 Re-Generate Patient-Level Manifests
```bash
python main.py prepare-rsna --seed 42
```
Re-partitions the 29,684 valid cohort into 70% Train, 15% Validation, and 15% Test grouped strictly by NIH patient ID.
