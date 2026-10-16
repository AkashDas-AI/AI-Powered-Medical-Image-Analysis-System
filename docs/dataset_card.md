# Dataset Card: RSNA Pneumonia Detection Challenge (2018)

## 1. Dataset Summary
- **Dataset Name**: RSNA Pneumonia Detection Challenge (2018)
- **Primary Domain**: Adult & Pediatric Thoracic Radiography
- **Target Diagnostic Task**: Binary Airspace Opacification Classification (`NO_LUNG_OPACITY` vs. `LUNG_OPACITY`)
- **Modality**: Frontal Chest Radiographs (Posteroanterior [PA] and Anteroposterior [AP] views)
- **Image Format**: Digital Imaging and Communications in Medicine (DICOM) Secondary Capture format (`.dcm`), 8-bit unsigned grayscale, $1024 \times 1024$ pixels
- **Volume**: 30,000 chest radiographs (16,248 PA views, 13,752 AP views) drawn from the 112,120-image National Institutes of Health (NIH) CXR8 corpus
- **Official Portal**: [RSNA 2018 Challenge Page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
- **Direct S3 Data Repository**: `https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/`
- **Original Challenge Publication**: Shih G, Wu CC, Halabi SS, Kohli MD, Prevedello LM, Cook TS, Sharma A, Amorosa JK, Arteaga V, Galperin-Aizenberg M, Gill RR, Godoy MC, Hobbs S, Jeudy J, Laroia A, Shah PN, Vummidi D, Yaddanapudi K, Stein A. *"Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia"*, Radiology: Artificial Intelligence 2019 1:1. DOI: [10.1148/ryai.2019180041](https://doi.org/10.1148/ryai.2019180041).
- **Foundation Dataset Publication**: Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases"*, IEEE CVPR, pp. 3462-3471, 2017. DOI: [10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369).

---

## 2. Cohort & Clinical Annotation Protocol
- **Cohort Provenance**: Subset of 30,000 frontal examinations randomly sampled from the NIH CXR8 dataset.
- **De-identification**: All Protected Health Information (PHI) was stripped by NIH and RSNA. Patient identifiers are anonymized hashes (`0000a175-0e68-4ca4-b1af-167204a7e0bc`). The first 8 digits of the original NIH filename (`XXXXXXXX_YYY.png`) map to the original de-identified patient index.
- **Expert Annotation Panel**:
  - Annotated using the `md.ai` platform by 18 board-certified radiologists from 16 medical institutions (including 12 chest radiologists from the Society of Thoracic Radiology [STR]).
  - Radiologists had an average clinical experience of 10.6 years (range: 3–35 years).
  - Annotations capture whole-study diagnostic categories and 2D bounding-box coordinates (`x`, `y`, `width`, `height`) around lung opacities suspicious for pneumonia.

---

## 3. Annotation Schema & Label Mapping Policy

The raw annotation payload contains 83,809 records across individual readers, adjudication rounds, and calculated consensus groups.

### 3.1 Adjudicated Consensus Categories
The ground-truth consensus (`Calculated` label group) partitions the 30,000 studies into three mutually exclusive categories:

| Adjudicated Category | md.ai Label ID | Study Count | Percent of Total | Bounding Boxes | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No Lung Opacity / Not Normal** | `L_yd0` | **12,788** | 43.08% | None | Radiographs exhibiting non-pneumonia thoracic abnormalities (e.g. cardiomegaly, pleural effusion, nodule, mass, atelectasis), but *no* parenchymal opacity suspicious for pneumonia. |
| **Normal** | `L_o8w` | **9,790** | 32.98% | None | Clear chest radiographs exhibiting no opacities or other thoracic abnormalities. |
| **Lung Opacity** | `L_v8n` | **7,106** | 23.94% | 11,390 ROIs | Radiographs exhibiting parenchymal opacification suspicious for pneumonia. |
| *Classified Subtotal* | — | *29,684* | *99.00%* | *11,390 ROIs* | Pairwise overlap between all 3 classes is **0**. |
| **Uncalculated / Excluded** | Multiple | **316** | 1.06% | Variable | Examinations flagged for exclusion by radiologists (`L_3lj`, `L_z94`, `L_kx3`: 120 cases) or lacking diagnostic consensus. |
| **Total Cohort** | — | **30,000** | **100.00%** | — | Complete official RSNA release. |

### 3.2 Label Mapping Policy for Binary Screening Task
In accordance with medical image screening best practices and the official RSNA/Kaggle challenge formulation:
- **`LUNG_OPACITY` (Class 1, Positive)**: Mapped from `Lung Opacity` ($n = 7,106$).
- **`NO_LUNG_OPACITY` (Class 0, Negative)**: Mapped from `Normal` ($n = 9,790$) **AND** `No Lung Opacity / Not Normal` ($n = 12,788$), totaling $22,578$ negative examinations.
- **Excluded Cases**: The 316 uncalculated/flagged records are strictly excluded from model training and evaluation to prevent label noise corruption.
- **Naming Rule**: Classes are explicitly named `LUNG_OPACITY` and `NO_LUNG_OPACITY` rather than `PNEUMONIA` and `NORMAL`, reflecting that lung opacity on radiography is an imaging finding rather than a microbiological diagnosis, and that non-opacity abnormalities are present in the negative cohort.

---

## 4. Patient-Level Splitting & Zero-Leakage Guarantee

### 4.1 Methodology
- Patient identifiers were derived from the official `pneumonia-challenge-dataset-mappings_2018.json` by parsing the 8-digit NIH patient identifier prefix (`XXXXXXXX_YYY.png`).
- Across the 29,684 classified examinations, there are exactly **12,233 unique patients** (7,464 single-image patients; 4,769 multi-image patients, with up to 75 images from a single patient).
- To prevent patient data leakage, splits are partitioned at the **patient level** using a deterministic random seed (`seed = 42`) with target proportions of 70% Train, 15% Validation, and 15% Test.

### 4.2 Split Manifest Breakdown
Manifests are serialized as CSV files under `data/processed/rsna/`:

| Split | Images | Unique Patients | `LUNG_OPACITY` (Positives) | `NO_LUNG_OPACITY` (Negatives) | Positive Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** (`train_manifest.csv`) | 20,913 | 8,563 (70.0%) | 5,094 | 15,819 | 24.36% |
| **Validation** (`val_manifest.csv`) | 4,370 | 1,835 (15.0%) | 982 | 3,388 | 22.47% |
| **Test** (`test_manifest.csv`) | 4,401 | 1,835 (15.0%) | 1,030 | 3,371 | 23.40% |
| **Total Valid Cohort** | **29,684** | **12,233** | **7,106** | **22,578** | **23.94%** |
| **Excluded** (`excluded_manifest.csv`) | 316 | 41 | — | — | — |

Programmatically asserted and verified:
$$\text{Train Patients} \cap \text{Val Patients} = \emptyset$$
$$\text{Train Patients} \cap \text{Test Patients} = \emptyset$$
$$\text{Val Patients} \cap \text{Test Patients} = \emptyset$$

---

## 5. Official Licensing & Attribution Terms

The dataset is governed by the official RSNA Terms of Use document (`pneumonia-challenge-terms-of-use-and-attribution.pdf`):

1. **Re-identification Prohibition**: Users agree not to make any attempt to identify or contact any individual(s) who may be subjects of the data.
2. **Attribution Requirements**:
   - Provide link to NIH CXR8 download site: `https://nihcc.app.box.com/v/ChestXray-NIHCC`
   - Acknowledge that the NIH Clinical Center is the primary data provider.
   - Provide link to RSNA challenge site: `https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018`
   - Include citations to Wang et al. (CVPR 2017) and Shih et al. (Radiology: AI 2019).

---

## 6. Known Limitations, Biases & Scientific Caveats

1. **Imaging Finding vs. Clinical Diagnosis**: Radiographic lung opacity correlates with pneumonia but is not pathognomonic. Bacterial pneumonia, viral pneumonia, pulmonary infarction, alveolar hemorrhage, and localized pulmonary edema can present with indistinguishable airspace opacification.
2. **Partial Local DICOM Availability**: The complete adjudicated DICOM archive is 3.71 GB (30,000 files). Local verification tests utilize verified extracted sample DICOM batches, while complete full-dataset training requires completing background archive transfer.
3. **Underlying NIH Text-Mining Provenance**: The NIH CXR8 baseline was indexed via NLP report mining, which carried an estimated label error rate of 10–15%. While the RSNA challenge panel re-adjudicated pneumonia opacities with expert chest radiologists, underlying co-morbidities derive from the original hospital corpus.
4. **Educational Prototype Status**: This codebase and documentation accompany an educational computer vision project. It is **not** a certified medical device, has not received regulatory clearance (FDA, CE-IVD), and must never be used for clinical diagnostic purposes.
