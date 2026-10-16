# Model Learning & Training Mechanics

This document provides a mathematical and procedural explanation of how models in this system learn, optimize their parameters, and generalize to unseen clinical data.

---

## 1. Learning Paradigms & Mathematical Formulations

### 1.1 Objective / Loss Functions

#### A. Classification: Focal Loss & Weighted Cross-Entropy
Medical datasets typically exhibit severe class imbalance (e.g., normal scans outnumber pathology scans). To penalize hard-to-classify samples and balance class frequencies:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t$ is the model's estimated probability for the true class.
- $\alpha_t \in [0, 1]$ addresses class imbalance.
- $\gamma \ge 0$ modulates focus toward difficult, easily-misclassified examples.

#### B. Segmentation: Combined Dice & Binary Cross Entropy (BCE) Loss
For spatial delineation of lesions:

$$\mathcal{L}_{\text{Combo}} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}$$

where the Dice Loss is given by:

$$\mathcal{L}_{\text{Dice}}(Y, \hat{Y}) = 1 - \frac{2 \sum_{i=1}^N Y_i \hat{Y}_i + \epsilon}{\sum_{i=1}^N Y_i + \sum_{i=1}^N \hat{Y}_i + \epsilon}$$

with $Y \in \{0, 1\}^N$ representing ground-truth binary mask voxels, $\hat{Y} \in [0, 1]^N$ representing predicted probabilities, and $\epsilon$ representing a smoothing factor preventing division by zero.

---

## 2. Optimization Strategy

### 2.1 Gradient Descent & Optimizers
- **Primary Optimizer**: AdamW (Decoupled Weight Decay Regularization).
  - Updates weights with separate decoupled weight decay:
    $$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
- **Alternative**: SGD with Nesterov momentum for fine-tuning pre-trained backbones.

### 2.2 Learning Rate Scheduling
- **Cosine Annealing with Warm Restarts / Warmup**:
  Gradual linear warmup for the first $E_{\text{warm}}$ epochs, followed by half-period cosine decay:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$

---

## 3. Generalization & Regularization Mechanisms

1. **Transfer Learning**: Pre-training on large-scale datasets (ImageNet / RadImageNet / Med3D) followed by staged fine-tuning (freezing shallow layers, fine-tuning deeper representations).
2. **Data Augmentation**: Spatial deformations (rotations, scaling, affine transforms), intensity jittering, and CutMix / MixUp to reduce over-reliance on local artifacts.
3. **Dropout & DropPath**: Stochastic depth in transformer blocks and standard dropout ($p \approx 0.2 - 0.4$) before dense projection heads.
4. **Mixed Precision (AMP)**: Automated Mixed Precision using FP16/BF16 to accelerate training and lower memory footprint without compromising numerical stability.

