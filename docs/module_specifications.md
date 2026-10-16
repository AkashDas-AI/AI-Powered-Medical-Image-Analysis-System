# Module Specifications & Component Blueprint

This document details the modular decomposition of the codebase. Each module is built with strict single-responsibility principles and decoupled interfaces to allow isolated testing and continuous extension.

---

## 1. Directory & Package Layout

```
src/
├── config/              # Centralized configuration management (YAML/Pydantic)
├── data/                # Dataset loaders, DICOM parsers, and transform pipelines
├── models/              # Neural network architectures, backbones, and heads
├── training/            # Training engine, optimizers, schedulers, and callbacks
├── evaluation/          # Metric computation, validation suite, and clinical benchmarks
├── explainability/      # Interpretability algorithms (Grad-CAM, saliency)
├── inference/           # Production inference engine and batch prediction
└── utils/               # Logging, I/O helpers, and visualization utilities
```

---

## 2. Core Modules Breakdown

### 2.1 Configuration Module (`src/config/`)
- **Primary Class**: `SystemConfig` / `ModelConfig` / `DataConfig`
- **Responsibility**: Loads hyperparameters, paths, hardware device configurations (CUDA/MPS/CPU), and logging setups via typed schemas. Ensures configuration reproducibility.

### 2.2 Data Module (`src/data/`)
- **Key Components**:
  - `MedicalDataset`: Base PyTorch dataset abstracting file resolution, label mapping, and on-the-fly caching.
  - `Preprocessor`: Implements intensity normalization, windowing, and channel formatting.
  - `DataModule`: Encapsulates train/validation/test split generation, worker thread pooling, and DataLoader instances.

### 2.3 Model Module (`src/models/`)
- **Key Components**:
  - `ModelFactory`: Factory pattern resolving requested architecture identifiers into initialized PyTorch modules.
  - `Backbone`: Transfer-learning feature extractors (e.g., ResNet, DenseNet, EfficientNet, ViT).
  - `SegmentationModel`: Encoder-decoder networks (e.g., U-Net, Attention U-Net) for mask generation.
  - `ClassificationModel`: Multi-class or multi-label classification networks.

### 2.4 Training & Optimization Module (`src/training/`)
- **Key Components**:
  - `Trainer`: Coordinates epoch execution, gradient backpropagation, mixed-precision (AMP), and validation sweeps.
  - `LossFactory`: Instantiates compound loss functions (e.g., Combo Loss = Binary Cross Entropy + Dice Loss).
  - `EarlyStopping` & `ModelCheckpoint`: Prevents overfitting and serializes best-performing model weights.

### 2.5 Explainability Module (`src/explainability/`)
- **Key Components**:
  - `GradCAM`: Gradient-weighted Class Activation Mapping module extracting feature activations and gradients from targeted convolutional layers.
  - `Visualizer`: Overlays normalized heatmaps onto raw grayscale/RGB medical scans.

### 2.6 Evaluation & Metrics Module (`src/evaluation/`)
- **Key Components**:
  - `ClassificationMetrics`: Computes Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, and AUROC.
  - `SegmentationMetrics`: Computes Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and Hausdorff Distance (HD95).
  - `ClinicalReporter`: Summarizes aggregate results and false positive/negative analysis.

