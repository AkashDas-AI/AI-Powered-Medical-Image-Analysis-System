# Technical Documentation Repository

Welcome to the central technical documentation hub for the AI-Powered Medical Image Analysis System. This directory hosts formal technical specifications, model cards, dataset governance documentation, and presentation materials.

---

## Technical Documentation Inventory

| Document | Primary Scope and Technical Content |
| :--- | :--- |
| [`architecture.md`](./architecture.md) | Modular system architecture, data layers, MobileNetV2 baseline, patient-level splitting flow, and memory-safe streaming design. |
| [`dataset_card.md`](./dataset_card.md) | RSNA 2018 adjudicated cohort provenance, 30,000 mapping records, 29,684 valid cohort, exclusion criteria, class distributions, and HIPAA de-identification. |
| [`model_card.md`](./model_card.md) | MobileNetV2 architecture specifications, training hyperparameters, held-out test benchmarks (AUROC 0.8238, Sensitivity 77.18%), decision thresholds, and calibration caveats. |
| [`limitations.md`](./limitations.md) | Detailed analysis of clinical, calibration, whole-image classification, adult-to-pediatric domain shift, and Grad-CAM interpretability constraints. |
| [`portfolio_demo.md`](./portfolio_demo.md) | Detailed 3 to 5 minute live demonstration walkthrough with reproducible PowerShell commands, expected console outputs, and visual checkpoint verifications. |
| [`portfolio_presentation.md`](./portfolio_presentation.md) | Comprehensive 12-slide portfolio and academic review presentation deck with detailed speaker notes, architectural breakdowns, and reviewer Q&A preparation. |

---

## Standards and Governance

1. Technical Rigor: All metrics reported across documentation match serialized evaluation records in `outputs/reports/` with zero synthetic performance fabrication.
2. Clinical Disclaimers: All documents maintain prominent regulatory notices that this system is an educational research prototype and is not FDA-cleared or clinically validated for diagnostic decision support.
3. Privacy and PHI: Documentation contains zero Protected Health Information (PHI). All sample outputs employ synthetic case identifiers.
