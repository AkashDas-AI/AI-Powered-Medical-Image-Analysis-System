# AI-Powered Medical Image Analysis System
## Final Portfolio & Academic Defense Presentation Package
### RSNA 2018 Adjudicated Chest Radiograph Opacity Classification & Explainability

---

> [!CAUTION]
> ### ⚠️ Mandatory Medical & Academic Research Disclaimer
> **This software is engineered strictly as an educational computer vision research prototype and student engineering portfolio project.**
> - It is **NOT** a certified, cleared, or approved medical diagnostic device (FDA, CE-MDR, etc.).
> - It **MUST NOT** be used for clinical diagnosis, patient screening, triage, or clinical management decisions.
> - Model scores represent **uncalibrated neural network feature activations**, NOT statistical disease likelihoods.
> - The target classification label is **radiographic lung opacity** (an anatomical imaging finding), NOT a definitive diagnosis of bacterial or viral pneumonia.

---

# Part 1: Presentation Slides & Defense Notes (12 Slides)

```
================================================================================
SLIDE 1: Title & Project Motivation
================================================================================
```

### AI-Powered Medical Image Analysis System
**Deep Transfer Learning & Explainability for Radiographic Lung Opacity**

- **Student Portfolio Presentation & Technical Defense**
- **Architecture**: MobileNetV2 Transfer Learning with Custom Classification Head
- **Dataset**: RSNA Pneumonia Detection Challenge 2018 (29,684 Adjudicated DICOMs)
- **Primary Focus**: End-to-end DICOM pipeline engineering, leak-free patient-level partitioning, memory-bounded streaming, gradient-weighted class activation mapping (Grad-CAM), and privacy-by-default software design.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 1):
"Good morning, members of the committee and reviewers. Today I am presenting my 
engineering portfolio project: an AI-Powered Medical Image Analysis System for 
radiographic lung opacity classification. 

Before diving into the architecture, I must state our mandatory research disclaimer: 
this project is engineered strictly for educational exploration and machine learning 
benchmarking. It is not an FDA-approved medical device, does not establish clinical 
diagnoses, and should never be used for patient management. 

Our core engineering goal was to design a production-grade, leak-free, and memory-
efficient computer vision pipeline capable of ingesting raw clinical DICOM images, 
extracting transparent feature attributions using Grad-CAM, and enforcing strict 
patient privacy safeguards by default."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 2: Problem Statement & Scope Definition
================================================================================
```

### Problem Context & Technical Framing

- **Clinical Problem**: Thoracic infections and lung consolidations are a leading cause of global respiratory illness. Chest radiography (CXR) is the ubiquitous primary imaging modality, but exhibits substantial inter-observer interpretation variability.
- **The Diagnostic Fallacy**: Machine learning papers frequently claim to "diagnose pneumonia" directly from radiographs. In clinical reality, a radiograph visualizes physical attenuation (tissue consolidation), not microbiological etiology.
- **Project Scope & Boundaries**:
  - **Educational Research Prototype**: Designed to demonstrate disciplined software architecture, defensive engineering, and transparent benchmarking.
  - **Explicit Target Label**: Binary classification of **visual radiographic lung opacity** adjudicated by thoracic radiologists, not automated microbiological diagnosis.
  - **Operational Constraint**: Full compatibility with consumer hardware; zero RAM spikes through bounded DICOM streaming.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 2):
"In machine learning literature, it is common to see claims of models 'diagnosing 
pneumonia' directly from X-ray pixels. From a medical imaging perspective, this is a 
critical fallacy. 

A radiograph displays physical photon attenuation—air spaces in the lung parenchyma 
filling with fluid, pus, blood, or collapsed tissue, which appear as white regions 
called 'opacities'. While an opacity may be caused by bacterial or viral pneumonia, 
it can equally stem from pulmonary edema, atelectasis, or infarction. 

Therefore, in this project, we explicitly scope our task to what the imaging data 
actually represents: the binary detection of expert-adjudicated radiographic lung 
opacity. We do not claim to diagnose disease, but rather evaluate how effectively a 
lightweight neural network can identify features correlated with expert radiologist 
consensus."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 3: RSNA 2018 Adjudicated Dataset & Label Semantics
================================================================================
```

### Dataset Provenance & Adjudication Standard

- **Official Source**: RSNA Pneumonia Detection Challenge (2018), staged by the Radiological Society of North America in collaboration with the Society of Thoracic Radiology (STR) and the National Institutes of Health (NIH).
- **Cohort Provenance**: 29,684 chest radiographs sampled from the NIH ChestX-ray14 repository.
- **Expert Adjudication**: Each case was reviewed by a panel of expert subspecialty thoracic radiologists who annotated visual regions of interest and established consensus binary labels:
  - **`NO_LUNG_OPACITY` (Class 0)**: Normal exams or radiographs containing non-opacity pathology (e.g., pneumothorax, nodules, scoliosis).
  - **`LUNG_OPACITY` (Class 1)**: Visual evidence of airspace consolidation or opacity consistent with pneumonia.
- **Cohort Class Distribution**:
  - `NO_LUNG_OPACITY`: 22,578 images (76.1%)
  - `LUNG_OPACITY`: 7,106 images (23.9%)
  - Natural Class Imbalance Ratio: **3.18 : 1**

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 3):
"We deliberately selected the RSNA 2018 dataset over web-scraped alternatives because 
of its rigorous adjudication standard. The dataset consists of 29,684 frontal chest 
radiographs derived from the NIH ChestX-ray14 cohort. 

Rather than relying on noisy NLP-extracted radiology reports, the RSNA partnered with 
the Society of Thoracic Radiology to have board-certified thoracic radiologists 
independently review and annotate every single image. 

The resulting cohort has 7,106 positive opacity cases and 22,578 non-opacity cases, 
representing an inherent class imbalance of approximately 3.2 to 1. Handling this 
imbalance and the original 16-bit DICOM files formed the foundation of our data 
engineering."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 4: Data Quality Audit & Leak-Free Patient Partitioning
================================================================================
```

### Preventing Data Contamination Across Splits

- **Data Integrity Audit**: Verified the 30,000 raw files against official mapping and annotation JSONs. Resolved 29,684 valid, adjudicated DICOMs; 316 cases were safely excluded with an audited tracking log (`excluded_manifest.csv`).
- **The Data Leakage Risk**: Multiple radiographs frequently belong to the same patient (taken at different study dates). Random image-level splitting causes identical patient anatomy to appear in both training and test sets, inflating metrics artificially.
- **Deterministic Cryptographic Partitioning**:
  - Computed deterministic hash: `SHA-256(patient_id + salt)`
  - Grouped all images strictly by unique patient identifier ($N = 12,233$ patients).
  - Partition target: 70% Train, 15% Validation, 15% Test.

| Split | DICOM Images | Unique Patients | No Lung Opacity | Lung Opacity | Patient Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 20,913 (70.4%) | 8,563 | 15,819 | 5,094 | **0** |
| **Validation** | 4,370 (14.7%) | 1,835 | 3,388 | 982 | **0** |
| **Held-Out Test** | 4,401 (14.8%) | 1,835 | 3,371 | 1,030 | **0** |

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 4):
"One of the most severe flaws in published medical computer vision benchmarks is data 
leakage. A single patient often has multiple radiographs across different clinical 
visits. If you split randomly at the image level, patient anatomy from the training set 
leaks into the test set, creating memorization rather than generalization.

To guarantee zero leakage, we extracted the true NIH Patient ID from each DICOM study 
and applied deterministic SHA-256 cryptographic hashing. 

As shown in the table, our 12,233 unique patients are partitioned into 8,563 for 
training, 1,835 for validation, and 1,835 for held-out testing. We have programmatically 
tested and verified that patient intersection across all splits is exactly zero."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 5: MobileNetV2 Architecture & Transfer Learning Design
================================================================================
```

### Lightweight Deep Learning Architecture

- **Backbone Selection**: **MobileNetV2** (ImageNet-pretrained convolutional feature extractor).
- **Engineering Rationale**:
  - Inverted residual bottleneck blocks with linear bottlenecks preserve manifold representations in low dimensions.
  - Depthwise separable convolutions reduce parameter count and computational complexity ($3.5\text{M}$ parameters vs. $25\text{M}$ in ResNet-50).
  - Enables low-latency CPU and GPU inference without requiring expensive clinical cluster hardware.
- **Custom Classification Head**:
  - Input: Preprocessed radiograph tensor $(224 \times 224 \times 3)$
  - Backbone Feature Extractor: Frozen base feature maps ending at layer `Conv_1` ($7 \times 7 \times 1280$)
  - Feature Compression: `GlobalAveragePooling2D` $\rightarrow$ $1280$-dimensional embedding vector
  - Regularization: `Dropout(rate=0.20)` to mitigate co-adaptation
  - Output Unit: Single `Dense(1, activation='sigmoid')` emitting continuous `lung_opacity_score` $\in [0.0, 1.0]$

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 5):
"For our modeling architecture, we selected MobileNetV2. In medical AI, there is often 
an unnecessary bias toward massive backbones like ResNet-152 or Vision Transformers. 
However, MobileNetV2 utilizes depthwise separable convolutions and inverted residual 
bottlenecks, giving us 3.5 million parameters with exceptional representational 
density. 

We froze the ImageNet-pretrained convolutional base to leverage low-level edge and 
texture filters. On top of the final convolutional layer—named Conv_1, which outputs 
a 7-by-7 feature grid with 1280 channels—we added a Global Average Pooling layer, a 
20% dropout regularization layer, and a single sigmoid dense unit. 

The output is a single continuous scalar between 0.0 and 1.0 representing the model's 
internal activation for radiographic lung opacity."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 6: DICOM Preprocessing Pipeline & Parity Guarantees
================================================================================
```

### Deterministic Multi-Step Image Normalization

```
Raw DICOM (.dcm) ──> pydicom Header & Pixel Decoding
                  ──> Photometric Inversion (MONOCHROME1 -> MONOCHROME2)
                  ──> Dynamic Min-Max Windowing (0 .. 255 uint8)
                  ──> PIL Bilinear RGB Resizing (224 x 224 x 3)
                  ──> MobileNetV2 Scaling: (x / 127.5) - 1.0 in [-1.0, 1.0]
```

1. **Photometric Interpretation**: In DICOM, `MONOCHROME1` defines zero as white (inverted polarity), whereas `MONOCHROME2` defines zero as black. The pipeline dynamically detects `MONOCHROME1` and performs mathematical inversion ($\max - x$) to align pixel intensities.
2. **Dynamic Windowing**: Clinical CXRs use 12-bit or 16-bit dynamic ranges. Min-max intensity scaling standardizes values to 8-bit $[0, 255]$ space.
3. **RGB Resizing**: Converts grayscale intensities to 3 identical channels and resizes via bilinear interpolation to $(224, 224, 3)$.
4. **MobileNetV2 Scaling**: Maps $[0, 255] \rightarrow [-1.0, 1.0]$ float32.
5. **Exact Pipeline Parity**: Verified with automated tests ensuring identical mathematical transformations across `RSNABatchGenerator`, `preprocess_image()`, and `predict_rsna_dicom()`.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 6):
"Medical DICOM images cannot simply be loaded with standard image libraries. A major 
engineering hurdle in this project was handling DICOM Photometric Interpretation. 

Some radiograph scanners encode images in MONOCHROME1, where zero is white and high 
values are black, while others use MONOCHROME2, where zero is black. If you do not 
invert MONOCHROME1 studies, lung tissue and bones are inverted, causing the model to 
fail completely. 

Our preprocessing pipeline inspects the DICOM metadata header, applies dynamic 
inversion when MONOCHROME1 is detected, performs min-max intensity windowing, resizes 
bilinearly to 224 by 224, and scales pixel floats into the minus 1.0 to plus 1.0 range 
required by MobileNetV2. 

Crucially, we wrote automated tests proving that training, batch evaluation, and 
single-study inference follow the exact same shared code path with bitwise parity."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 7: Training Setup, Class Imbalance, & One-Epoch Boundary
================================================================================
```

### Reproducible Training & Resource Constraints

- **Streaming Batch Generator**: `RSNABatchGenerator` inherits from Keras `Sequence` / `PyDataset`. It lazily loads DICOMs from disk batch-by-batch (batch size 16), ensuring RAM consumption remains flat regardless of dataset scale.
- **Class Imbalance Strategy**: Applied inverse class frequency weighting derived strictly from the training cohort:
  $$w_0 = \frac{N}{2 \cdot N_0} = \frac{20913}{2 \cdot 15819} \approx 0.6610, \quad w_1 = \frac{N}{2 \cdot N_1} = \frac{20913}{2 \cdot 5094} \approx 2.0527$$
  Positive opacity samples receive $\approx 3.1\times$ higher gradient update weight during backpropagation.
- **Optimizer & Schedule**: Adam optimizer ($\text{lr} = 10^{-4}$), binary cross-entropy loss.
- **The One-Epoch Baseline Boundary**:
  - The model was trained for **1 full pass** across all 20,913 training images (1,308 steps) and validated over all 4,370 validation images (274 steps).
  - **Academic Rigor**: Multi-epoch convergence was not established. This checkpoint serves as a verified, leak-free transfer-learning baseline.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 7):
"Training on 29,684 DICOM files would easily crash standard consumer hardware if loaded 
into system RAM. We engineered a bounded, streaming batch generator that reads, 
normalizes, and yields batches of 16 images on-the-fly. 

To counteract the 3.2-to-1 class imbalance without synthetic data corruption, we computed 
balanced class weights strictly from the training split: class 0 received a weight of 
0.66, while positive opacity received a weight of 2.05.

Here I must highlight an essential limitation: our active checkpoint completed exactly 
one full pass over the 20,913 training images and 4,370 validation images. We did not 
claim multi-epoch convergence. This experiment established an audited transfer-learning 
baseline, demonstrating what ImageNet feature representations achieve after a single 
pass over thoracic data."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 8: Held-Out Test Split Benchmark Evaluation
================================================================================
```

### Quantitative Screening Metrics ($N = 4,401$ Test Images)

- **Evaluation Protocol**: Evaluated strictly on the held-out test split (1,835 unseen patients) at a pre-specified research threshold of $\tau = 0.50$ without post-hoc test tuning.

| Metric | Measured Value | Standard Formula | Clinical & Reviewer Interpretation |
| :--- | :---: | :---: | :--- |
| **AUROC** | **0.8238** | Area under ROC | Strong discriminative ability independent of decision threshold. |
| **Accuracy** | **74.91%** | $\frac{TP + TN}{N}$ | 3,297 out of 4,401 correct predictions on held-out patients. |
| **Sensitivity (Recall)** | **77.18%** | $\frac{TP}{TP + FN}$ | Successfully detected 795 of 1,030 radiologist-confirmed opacities. |
| **Specificity** | **74.22%** | $\frac{TN}{TN + FP}$ | Correctly ruled out 2,502 of 3,371 non-opacity studies. |
| **Precision (PPV)** | **47.78%** | $\frac{TP}{TP + FP}$ | 795 of 1,664 positive predictions confirmed by radiologist truth. |
| **F1-Score** | **0.5902** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean balancing recall and precision under imbalance. |

### Held-Out Test Confusion Matrix
$$\begin{pmatrix} TN & FP \\ FN & TP \end{pmatrix} = \begin{pmatrix} 2,502 & 869 \\ 235 & 795 \end{pmatrix}$$
- **Understanding Precision (47.78%)**: In medical screening, positive predictive value is heavily influenced by cohort prevalence (23.4% in test set). Prioritizing high recall (77.2%) minimizes missed pathologies (FN = 235).

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 8):
"Let us examine the quantitative benchmark on our 4,401 held-out test cases. 
At our fixed benchmark threshold of 0.50, the model achieved an Area Under the ROC 
Curve of 0.8238, an overall accuracy of 74.91%, and a sensitivity of 77.18%. 

A key question reviewers often ask is: 'Why is precision 47.78% when accuracy is 75%?' 
The answer lies in epidemiology and Bayes' theorem. Precision, or Positive Predictive 
Value, depends heavily on disease prevalence. In our held-out test cohort, only 23.4% 
of images have opacities. 

Because we used inverse class weighting to penalize false negatives, the model acts as 
a sensitive screening tool: it identified 795 out of 1,030 true opacities, keeping false 
negatives down to 235, at the cost of 869 false positives."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 9: End-to-End Inference Workflow & Real Cases
================================================================================
```

### Production-Grade CLI Inference & Privacy Redaction

```powershell
python main.py predict-rsna --image "<path.dcm>" --threshold 0.50 --gradcam `
    --output "outputs/plots/gradcam_case_opacity.png" `
    --json-output "outputs/reports/prediction_case_opacity.json"
```

#### Real Study Comparisons:
```
Case 1: Normal CXR Study (Negative)         Case 2: Lung Opacity CXR Study (Positive)
----------------------------------------    ----------------------------------------
Modality:            CR                     Modality:            CR
View Position:       PA                     View Position:       AP
Photometric Interp:  MONOCHROME2            Photometric Interp:  MONOCHROME2
Dimensions:          1024 x 1024            Dimensions:          1024 x 1024
Model Score:         0.1154                 Model Score:         0.7101
Decision Threshold:  0.50                   Decision Threshold:  0.50
Classification:      NO_LUNG_OPACITY        Classification:      LUNG_OPACITY
Opacity Detected:    NO (Negative)          Opacity Detected:    YES (Positive)
Probabilities:       0.8846 vs. 0.1154      Probabilities:       0.2899 vs. 0.7101
```
- **Privacy-by-Default Contract**: Patient ID, StudyInstanceUID, and SOPInstanceUID are stripped from stdout and JSON reports. Only non-sensitive technical metadata is displayed.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 9):
"Here we see our end-to-end inference workflow executed on real, verified RSNA DICOM 
studies. 

On the left is a normal PA radiograph. The model evaluates the tensor, emits an 
uncalibrated continuous score of 0.1154, and correctly classifies it as NO_LUNG_OPACITY. 

On the right is an AP radiograph exhibiting prominent bilateral infiltrates. The model 
outputs a score of 0.7101, successfully classifying it as LUNG_OPACITY. 

Notice that in both outputs, zero protected health information is revealed. Patient 
names, Patient IDs, and hospital Study UIDs are stripped by default. Only technical 
parameters like imaging modality, view position, and image matrix dimensions are logged."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 10: Grad-CAM Explainability & Saliency Limitations
================================================================================
```

### Convolutional Feature Attribution & Interpretability

- **Grad-CAM Mechanics**: Computes the gradient of the lung opacity score $y$ with respect to feature maps $A^k$ at layer `Conv_1` ($7 \times 7 \times 1280$):
  $$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
- **Structured 3-Panel Visualization**:
  - **Panel 1**: Preprocessed chest radiograph (normalized with MONOCHROME inversion).
  - **Panel 2**: Normalized feature saliency map ($0.0 = \text{low}$, $1.0 = \text{high}$) with quantitative colorbar legend.
  - **Panel 3**: Blended overlay ($\alpha = 0.40$) mapping spatial activation over thoracic anatomy.
- **Degenerate Attribution Guard**: If positive gradients are zero ($\max = 0$), the pipeline suppresses deceptive synthetic blue colormaps and renders the uncolored radiograph labeled `[Zero Positive Attribution / Degenerate]`.
- **Interpretability Limitation**: Grad-CAM visualizes **coarse network feature correlation** ($7 \times 7$ upsampled to $224 \times 224$), NOT anatomical pathology margins or microbiological boundaries.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 10):
"To provide transparency into model behavior, we implemented Gradient-Weighted Class 
Activation Mapping, or Grad-CAM. We hook into the final convolutional layer, Conv_1, 
calculate the gradients of the opacity score with respect to all 1280 feature channels, 
globally pool the gradients, and compute a rectified weighted sum. 

Our visualization displays three panels: the preprocessed radiograph, the normalized 
saliency map with an explicit colorbar legend, and an anatomical overlay. 

We also engineered a critical safety guard: in edge cases where positive gradients are 
zero, standard Grad-CAM implementations produce a misleading all-blue colormap wash. 
Our code detects this degenerate state and outputs the uncolored base image with an 
explicit notification. 

We must emphasize to clinicians and reviewers that Grad-CAM heatmaps represent feature 
correlations, not radiological proof of lesion boundaries."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 11: Privacy Engineering, Software Testing, & Reproducibility
================================================================================
```

### Production-Grade Software Engineering Discipline

- **Privacy-by-Default Architecture**:
  - Medical DICOMs embed institutional and patient UIDs.
  - Defaulting `include_patient_id=False` in code and CLI eliminates accidental PHI leakage.
  - Opting in via `--include-patient-id` is restricted to authorized audits and emits explicit privacy warnings.
- **Git Hygiene**:
  - Strict `.gitignore` excludes raw datasets (`data/raw/`), processed caches (`data/processed/`), model weights (`models/*.keras`), and generated reports/plots (`outputs/`).
  - Confirmed via `git check-ignore` and `git status --ignored`.
- **Comprehensive Automated Regression Suite**:
  - **76 Automated Tests** passing across unit, integration, and CLI modules with **0 warnings**:
    - Preprocessing parity across DICOM/JPEG and bitwise equality verification
    - Inversion parity across `MONOCHROME1` and `MONOCHROME2`
    - Patient-level leak-free hash partitioning
    - Defensive CLI handling (out-of-bounds thresholds, missing files, corrupt DICOMs)
    - Finite Grad-CAM tensor validation and degenerate attribution suppression

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 11):
"Beyond deep learning algorithms, this project was developed under strict software 
engineering and quality assurance standards. 

Medical software must handle privacy defensively. Our pipeline excludes patient 
identifiers by default from all logs, stdout, and JSON exports. Our Git repository 
is configured so that no patient DICOMs, model weights, or output plots can ever be 
accidentally committed. 

Furthermore, our test suite contains 76 automated pytest tests that execute with zero 
warnings. These tests cover everything from bitwise preprocessing parity and 
MONOCHROME1 inversion math, to corrupt DICOM handling and boundary condition checks on 
CLI arguments. 

Every single claim in our documentation is backed by automated tests and verifiable 
disk artifacts."
--------------------------------------------------------------------------------
```

---

```
================================================================================
SLIDE 12: Project Limitations, Future Roadmap, & Defense Summary
================================================================================
```

### Critical Reflection & Research Roadmap

- **Documented Limitations**:
  1. **Single-Epoch Scope**: Active checkpoint was trained for 1 complete epoch; multi-epoch convergence and learning-rate decay schedules remain unestablished.
  2. **Uncalibrated Probabilities**: Sigmoid activations reflect feature correlation, not calibrated clinical risk probabilities (temperature scaling / Platt scaling needed).
  3. **Classification vs. Localization**: The model classifies whole-image opacity; it does not predict explicit bounding boxes or segmentation masks.
  4. **Domain Shift**: Trained on adult radiographs from NIH CXR-14; pediatric transferability requires domain-specific fine-tuning.
- **Technical Roadmap**:
  - Multi-epoch unfreezing of MobileNetV2 top blocks with cosine decay.
  - Bounding-box detection using YOLO or RetinaNet trained on RSNA bounding box coordinates.
  - Expected Calibration Error (ECE) measurement and isotonic regression calibration.
- **Defense Summary**:
  - Implemented an audited, leak-free, reproducible transfer-learning baseline.
  - Achieved **0.8238 AUROC** and **77.18% sensitivity** on 4,401 held-out test DICOMs.
  - Complete code transparency, comprehensive testing, and strict privacy controls.

```
--------------------------------------------------------------------------------
SPEAKER NOTES (Slide 12):
"To conclude, let us review our technical limitations and future roadmap. 

First, our baseline completed one full training epoch. In future work, we plan to unfreeze 
the top convolutional blocks and execute multi-epoch fine-tuning with cosine annealing. 
Second, our sigmoid outputs require formal post-hoc calibration, such as temperature 
scaling, before probabilities could reflect true empirical risk. 
Third, moving from whole-image classification to bounding-box detection using the RSNA 
bounding box coordinates is a natural next step. 

In summary, this project delivers a disciplined, leak-free, and privacy-preserving 
educational computer vision baseline. It demonstrates how to handle raw clinical DICOM 
data, manage class imbalance, explain convolutional representations, and engineer 
reproducible deep learning software. 

Thank you for your time, and I welcome any questions from the committee."
--------------------------------------------------------------------------------
```

---

# Part 2: Live Demonstration Sequence (3–5 Minutes)

This step-by-step walkthrough is engineered for live execution during an academic review, viva, or portfolio presentation. All commands use relative paths and standard Windows PowerShell syntax.

### Prerequisites: Terminal Setup
```powershell
# 1. Open PowerShell and navigate to the project directory
cd "D:\Projects\Artificial Intelligence\AI-Powered Medical Image Analysis System"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

---

### Step 1: Demonstrate Codebase Architecture & Integrity (30 seconds)
Show the audience the modular structure and clean repository layout:

```powershell
# View core directory structure
Get-ChildItem -Directory | Select-Object Name
```
- **Talking Point**: *"The codebase is structured into strict modules: `src/` for core logic, `data/` for dataset manifests, `models/` for checkpoints, `outputs/` for reports, and `tests/` for our automated test suite."*

---

### Step 2: Run a Basic DICOM Prediction (Privacy-Preserving) (45 seconds)
Execute forward inference on a verified normal DICOM study:

```powershell
.\.venv\Scripts\python.exe main.py predict-rsna `
    --image "data/raw/rsna/dicom/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/1.2.276.0.7230010.3.1.2.8323329.1475.1517874291.115797/1.2.276.0.7230010.3.1.3.8323329.1475.1517874291.115796/1.2.276.0.7230010.3.1.4.8323329.1475.1517874291.115798.dcm"
```
- **Observed Result**: Output prints `Model-Estimated Score: 0.1154`, `Classification: NO_LUNG_OPACITY`, and `Modality: CR`.
- **Talking Point**: *"Notice the fast inference latency on standard CPU hardware. Notice also that Patient ID and hospital study UIDs are completely absent from stdout, enforcing privacy-by-default."*

---

### Step 3: Run Inference with Grad-CAM & JSON Export (60 seconds)
Execute forward inference on a positive lung opacity DICOM study, generating a multi-panel Grad-CAM visualization and exporting structured JSON:

```powershell
.\.venv\Scripts\python.exe main.py predict-rsna `
    --image "data/raw/rsna/dicom/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/1.2.276.0.7230010.3.1.2.8323329.1476.1517874291.106014/1.2.276.0.7230010.3.1.3.8323329.1476.1517874291.106013/1.2.276.0.7230010.3.1.4.8323329.1476.1517874291.106015.dcm" `
    --threshold 0.50 `
    --gradcam `
    --output "outputs/plots/gradcam_case_opacity.png" `
    --json-output "outputs/reports/prediction_case_opacity.json"
```
- **Observed Result**: Prints `Model-Estimated Score: 0.7101`, `Classification: LUNG_OPACITY`, and confirms successful serialization of visualization and JSON report.

---

### Step 4: Inspect Generated Explainability & Report Artifacts (45 seconds)
Open and display the generated artifacts to the audience:

```powershell
# View structured JSON report
Get-Content outputs/reports/prediction_case_opacity.json

# Open Grad-CAM multi-panel visualization in default image viewer
Invoke-Item outputs/plots/gradcam_case_opacity.png
```
- **Talking Point**: *"Here we see our structured JSON report, containing only non-sensitive technical metadata and class probabilities. The Grad-CAM figure shows the preprocessed radiograph, the normalized saliency heatmap from layer Conv_1, and the blended overlay with a calibrated colorbar."*

---

### Step 5: Demonstrate Software Reliability & Defensive Error Handling (30 seconds)
Demonstrate defensive validation on invalid threshold and missing file inputs:

```powershell
# Out-of-bounds threshold validation (exits with code 1)
.\.venv\Scripts\python.exe main.py predict-rsna --image "<path.dcm>" --threshold 1.5

# Missing file validation (exits with code 1)
.\.venv\Scripts\python.exe main.py predict-rsna --image "data/raw/rsna/dicom/missing.dcm"
```
- **Talking Point**: *"The CLI is defensively programmed: passing out-of-bounds thresholds or non-existent files immediately exits with a descriptive error and non-zero exit code without unhandled stack traces."*

---

### Step 6: Verify Regression Test Suite & Benchmark Evidence (45 seconds)
Execute the complete test suite to prove regression robustness:

```powershell
# Run the complete project test suite
.\.venv\Scripts\python.exe -m pytest -q
```
- **Observed Result**: `76 passed in ~69s` with **0 warnings**.
- **Talking Point**: *"All 76 unit and integration tests pass with zero warnings, verifying leak-free hashing, preprocessing parity, DICOM decoding, and Grad-CAM correctness."*

---

# Part 3: Reviewer Q&A Reference Guide

### Q1: Why did you label the task "lung opacity detection" rather than "pneumonia detection"?
> **Response**: "In clinical chest radiography, pneumonia presents as an airspace opacity (consolidation or infiltrates). However, an opacity is an anatomical imaging appearance, not a microbiological diagnosis. Non-infectious conditions such as pulmonary edema, atelectasis, and pulmonary hemorrhage can produce identical opacities. The RSNA 2018 challenge ground truth was annotated by thoracic radiologists identifying visual opacities consistent with pneumonia. To maintain scientific integrity and avoid overpromising clinical capability, we describe the model strictly by what it actually predicts: radiographic lung opacity."

### Q2: What evidence guarantees that no patient data leaked between training and test sets?
> **Response**: "Patients frequently have multiple radiographs in the RSNA dataset. If we had partitioned images randomly, radiographs from the same patient would appear in both training and test sets. We extracted the true NIH patient identifier, applied a deterministic cryptographic SHA-256 hash combined with a fixed random seed (42), and partitioned strictly by patient ID. In our test suite, `test_patient_leak_free_split` programmatically checks the intersection of patient IDs across train, validation, and test sets, verifying that patient overlap is exactly zero."

### Q3: Why is precision (PPV) 47.78% when overall accuracy is 74.91%?
> **Response**: "Precision is heavily dependent on class prevalence. In our held-out test cohort, positive opacity cases represent only 23.4% of the population (1,030 out of 4,401 cases). Because we incorporated inverse class weighting during training to prioritize detecting true positives, the model operates with high sensitivity (77.18%), identifying 795 of 1,030 opacity cases while keeping false negatives to 235. In medical screening contexts, a false negative (missing an active consolidation) is clinically far more critical than a false positive (which can be reviewed and ruled out by a radiologist)."

### Q4: Can Grad-CAM heatmaps be used to delineate exact pathological margins?
> **Response**: "No. Grad-CAM visualizes the feature maps of MobileNetV2's final convolutional layer, `Conv_1`, which has a spatial resolution of $7 \times 7$. Upsampling a $7 \times 7$ grid to $224 \times 224$ produces a coarse regional heatmap showing which receptive fields contributed positively to the classifier's activation. It indicates regional feature correlation, not anatomical segmentation or microscopic pathology boundaries."

### Q5: Why did the baseline training stop after one epoch?
> **Response**: "The full RSNA training split contains 20,913 images, which requires 1,308 gradient update steps per epoch at batch size 16. The primary objective of this phase was to construct an audited, leak-free, full-pass baseline and verify our streaming DICOM generator without memory leaks. We explicitly document this limitation: multi-epoch convergence and learning-rate decay schedules are not claimed and represent future work."

### Q6: How do you safeguard patient privacy in this project?
> **Response**: "Clinical DICOM headers embed protected health information (PHI) including Patient ID, StudyInstanceUID, and SOPInstanceUID. Our system implements Privacy-by-Default: single-image inference and JSON serialization functions exclude all sensitive patient and study identifiers by default. Opting into patient IDs requires an explicit `--include-patient-id` flag intended only for authenticated local auditing. Furthermore, our `.gitignore` strictly prevents raw DICOMs, model weights, and local reports from ever being committed to Git."

