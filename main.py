"""Command-Line Interface (CLI) for the AI-Powered Medical Image Analysis System.

Provides lightweight, modular subcommands for auditing datasets, training models,
evaluating test benchmarks, running single-image inference, and generating
exploratory Grad-CAM visual heatmaps.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    PLOTS_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RSNA_BEST_MODEL_PATH,
    RSNA_DICOM_DIR,
    RSNA_FINAL_MODEL_PATH,
    RSNA_METADATA_DIR,
    RSNA_PROCESSED_DIR,
    RSNA_TEST_MANIFEST,
    RSNA_TRAIN_MANIFEST,
    RSNA_VAL_MANIFEST,
    SMOKE_TEST_MODEL_PATH,
)
from src.data_audit import audit_dataset, format_audit_summary


def create_parser() -> argparse.ArgumentParser:
    """Construct and configure the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ai-medical-analysis",
        description="AI-Powered Medical Image Analysis System (Pediatric Pneumonia Detection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="Available Subcommands",
        description="Select a pipeline task to execute",
        required=True,
    )

    # 1. Audit Subcommand
    audit_parser = subparsers.add_parser(
        "audit",
        help="Inspect dataset directory layout, verify splits, and count image files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    audit_parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Path to the raw dataset directory to audit.",
    )

    # 2. Train Subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Train the MobileNetV2 pneumonia classification model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Root dataset directory containing train/ and val/ splits.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum training epochs.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for Adam optimizer.",
    )
    train_parser.add_argument(
        "--save-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save the trained model (.keras).",
    )

    # 3. Evaluate Subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model against the test dataset and compute clinical metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the serialized model file (defaults to best available in models/).",
    )
    eval_parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to the test data split (defaults to standard raw test folder).",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification decision threshold for positive class.",
    )

    # 4. Predict Subcommand
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference on a single chest X-ray image with optional Grad-CAM explainability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the chest X-ray image (JPEG/PNG).",
    )
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint (defaults to best available in models/).",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold for Pneumonia classification.",
    )
    predict_parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate and save an exploratory Grad-CAM saliency map overlay.",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the Grad-CAM visualization or prediction result JSON.",
    )

    # 5. Audit RSNA Subcommand
    rsna_audit_parser = subparsers.add_parser(
        "audit-rsna",
        help="Audit RSNA Pneumonia Detection Challenge metadata, DICOM files, and manifest integrity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_audit_parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(RSNA_METADATA_DIR),
        help="Directory containing RSNA mapping and annotation JSON files.",
    )
    rsna_audit_parser.add_argument(
        "--dicom-dir",
        type=str,
        default=str(RSNA_DICOM_DIR),
        help="Directory containing extracted RSNA DICOM images.",
    )
    rsna_audit_parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(RSNA_PROCESSED_DIR),
        help="Directory containing generated RSNA manifests.",
    )
    rsna_audit_parser.add_argument(
        "--max-checks",
        type=int,
        default=None,
        help="Optional maximum number of local DICOM files to test for pixel readability.",
    )

    # 6. Prepare RSNA Manifests Subcommand
    rsna_prep_parser = subparsers.add_parser(
        "prepare-rsna",
        help="Generate leak-free, patient-level train/val/test manifests for RSNA cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_prep_parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(RSNA_METADATA_DIR),
        help="Directory containing RSNA mapping and annotation JSON files.",
    )
    rsna_prep_parser.add_argument(
        "--dicom-dir",
        type=str,
        default=str(RSNA_DICOM_DIR),
        help="Directory containing RSNA DICOM images.",
    )
    rsna_prep_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RSNA_PROCESSED_DIR),
        help="Output directory to save processed manifests.",
    )
    rsna_prep_parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible patient-level splitting.",
    )

    # 7. Download RSNA Subcommand
    rsna_download_parser = subparsers.add_parser(
        "download-rsna",
        help="Download or resume the official RSNA adjudicated DICOM archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_download_parser.add_argument(
        "--target-path",
        type=str,
        default=r"D:\tmp_rsna\pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip",
        help="Path on D: to save the downloaded archive.",
    )
    rsna_download_parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=4,
        help="Size of range chunks in MB (default: 4 MB).",
    )
    rsna_download_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent range worker threads (default: 4).",
    )
    rsna_download_parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum download duration in seconds before pausing cleanly.",
    )
    rsna_download_parser.add_argument(
        "--extract",
        action="store_true",
        help="Automatically verify and safely extract archive upon completion.",
    )
    # 8. RSNA Smoke Test Subcommand
    rsna_smoke_parser = subparsers.add_parser(
        "smoke-test-rsna",
        help="Run an end-to-end forward/backward smoke test on MobileNetV2 with RSNA DICOM batches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_smoke_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for smoke testing.",
    )
    rsna_smoke_parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        help="Backbone pretrained weights ('imagenet' or 'none').",
    )
    rsna_smoke_parser.add_argument(
        "--train-manifest",
        type=str,
        default=str(RSNA_TRAIN_MANIFEST),
        help="Path to training manifest CSV.",
    )
    rsna_smoke_parser.add_argument(
        "--dicom-dir",
        type=str,
        default=str(RSNA_DICOM_DIR),
        help="Root directory of RSNA DICOM images.",
    )

    # 9. RSNA Train Subcommand
    rsna_train_parser = subparsers.add_parser(
        "train-rsna",
        help="Train the MobileNetV2 transfer-learning baseline on RSNA adjudicated chest X-rays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum training epochs.",
    )
    rsna_train_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for gradient updates.",
    )
    rsna_train_parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate for Adam optimizer.",
    )
    rsna_train_parser.add_argument(
        "--freeze-base",
        dest="freeze_base",
        action="store_true",
        default=True,
        help="Freeze MobileNetV2 backbone parameters.",
    )
    rsna_train_parser.add_argument(
        "--no-freeze-base",
        dest="freeze_base",
        action="store_false",
        help="Unfreeze MobileNetV2 backbone parameters for end-to-end fine-tuning.",
    )
    rsna_train_parser.add_argument(
        "--weights",
        type=str,
        default="imagenet",
        help="Pretrained weights ('imagenet' or 'none').",
    )
    rsna_train_parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Number of training steps per epoch (default: all manifest batches).",
    )
    rsna_train_parser.add_argument(
        "--val-steps",
        type=int,
        default=None,
        help="Number of validation steps per epoch (default: all manifest batches).",
    )
    rsna_train_parser.add_argument(
        "--train-manifest",
        type=str,
        default=str(RSNA_TRAIN_MANIFEST),
        help="Path to RSNA training manifest CSV.",
    )
    rsna_train_parser.add_argument(
        "--val-manifest",
        type=str,
        default=str(RSNA_VAL_MANIFEST),
        help="Path to RSNA validation manifest CSV.",
    )
    rsna_train_parser.add_argument(
        "--dicom-dir",
        type=str,
        default=str(RSNA_DICOM_DIR),
        help="Directory containing RSNA DICOM images.",
    )
    rsna_train_parser.add_argument(
        "--save-path",
        type=str,
        default=str(RSNA_FINAL_MODEL_PATH),
        help="Destination path for final trained model.",
    )
    rsna_train_parser.add_argument(
        "--best-path",
        type=str,
        default=str(RSNA_BEST_MODEL_PATH),
        help="Destination path for best validation model checkpoint.",
    )
    rsna_train_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory to save training configuration and metrics CSV.",
    )
    rsna_train_parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Directory to save training plots.",
    )

    # 10. RSNA Evaluate Subcommand
    rsna_eval_parser = subparsers.add_parser(
        "evaluate-rsna",
        help="Evaluate trained model on the held-out RSNA test split and compute clinical benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_eval_parser.add_argument(
        "--model-path",
        type=str,
        default=str(RSNA_BEST_MODEL_PATH),
        help="Path to trained model checkpoint.",
    )
    rsna_eval_parser.add_argument(
        "--test-manifest",
        type=str,
        default=str(RSNA_TEST_MANIFEST),
        help="Path to held-out RSNA test manifest CSV.",
    )
    rsna_eval_parser.add_argument(
        "--dicom-dir",
        type=str,
        default=str(RSNA_DICOM_DIR),
        help="Directory containing RSNA DICOM images.",
    )
    rsna_eval_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Operational classification decision threshold.",
    )
    rsna_eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for test evaluation inference.",
    )
    rsna_eval_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for quick test verification.",
    )
    rsna_eval_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory to save evaluation report JSON.",
    )
    rsna_eval_parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Directory to save evaluation plots.",
    )

    # 11. Predict RSNA Subcommand
    rsna_predict_parser = subparsers.add_parser(
        "predict-rsna",
        help="Run inference on a single RSNA DICOM chest X-ray with optional Grad-CAM explainability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rsna_predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the RSNA DICOM (.dcm) image.",
    )
    rsna_predict_parser.add_argument(
        "--model-path",
        type=str,
        default=str(RSNA_BEST_MODEL_PATH),
        help="Path to the trained RSNA model checkpoint.",
    )
    rsna_predict_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Operational decision threshold for lung opacity classification.",
    )
    rsna_predict_parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate and save an exploratory Grad-CAM saliency map overlay.",
    )
    rsna_predict_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the Grad-CAM visualization image (explicitly overwrites if file exists).",
    )
    rsna_predict_parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="Target convolutional feature layer for Grad-CAM (default: auto-detected Conv_1).",
    )
    rsna_predict_parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to save structured prediction result JSON (explicitly overwrites if file exists).",
    )
    rsna_predict_parser.add_argument(
        "--include-patient-id",
        action="store_true",
        help="Include de-identified Patient ID in prediction output (opt-in for debugging/audit). Caution: Do not enable when sharing logs or reports publicly.",
    )

    return parser


def run_audit(args: argparse.Namespace) -> int:
    """Execute the dataset audit subcommand."""
    print(f"\n[AUDIT] Auditing dataset directory: {args.data_dir} ...")
    report = audit_dataset(args.data_dir)
    print(format_audit_summary(report))
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Execute the model training subcommand."""
    from src.train import train_model

    print("\n[TRAIN] Initiating model training workflow...")
    print(f"Dataset: {args.data_dir} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    try:
        result = train_model(
            data_dir=args.data_dir,
            model_save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        print("\n[SUCCESS] Training Complete!")
        print(f"Final Model Saved: {result['model_path']}")
        print(f"Best Model Saved:  {result['best_model_path']}")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Training failed: {exc}", file=sys.stderr)
        return 1


def run_evaluate(args: argparse.Namespace) -> int:
    """Execute the model evaluation subcommand."""
    from src.evaluate import evaluate_model

    print(f"\n[EVAL] Evaluating model '{args.model_path}' ...")
    try:
        metrics = evaluate_model(
            model_path=args.model_path,
            test_data_dir=args.test_dir,
            threshold=args.threshold,
        )
        print("\nClinical Benchmark Metrics:")
        print(json.dumps(metrics, indent=2))
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        return 1


def run_predict(args: argparse.Namespace) -> int:
    """Execute single image prediction with optional Grad-CAM."""
    from src.predict import load_inference_model, predict_image
    from src.preprocessing import preprocess_for_model

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}", file=sys.stderr)
        return 1

    # Delegate DICOM files directly to the RSNA prediction workflow
    if image_path.suffix.lower() == ".dcm":
        return run_predict_rsna(args)

    print(f"\n[PREDICT] Running inference on: {image_path} ...")
    try:
        model = load_inference_model(args.model_path)
        result = predict_image(
            image_input=image_path,
            model=model,
            threshold=args.threshold,
        )

        print("\n--- Diagnostic Prediction Output ---")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence'] * 100:.2f}%")
        print(f"Raw Activation:  {result['raw_score']:.4f}")
        print(f"Probabilities:   NORMAL: {result['probabilities']['NORMAL']:.4f} | PNEUMONIA: {result['probabilities']['PNEUMONIA']:.4f}")
        print(f"\n[DISCLAIMER] {result['disclaimer']}")

        # Handle optional Grad-CAM
        if args.gradcam:
            from src.explainability import generate_gradcam, overlay_heatmap

            print("\n[EXPLAIN] Generating Grad-CAM saliency map...")
            tensor = preprocess_for_model(image_path)
            heatmap = generate_gradcam(model, tensor)
            overlay = overlay_heatmap(image_path, heatmap)

            output_plot_path = (
                Path(args.output)
                if args.output
                else PLOTS_DIR / f"gradcam_{image_path.stem}.png"
            )
            output_plot_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(output_plot_path)
            print(f"[SUCCESS] Saliency map saved to: {output_plot_path}")

        return 0
    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}", file=sys.stderr)
        return 1


def run_predict_rsna(args: argparse.Namespace) -> int:
    """Execute single RSNA DICOM image prediction with optional Grad-CAM explainability."""
    import json
    import numpy as np
    from src.predict import load_inference_model, predict_rsna_dicom
    from src.explainability import generate_gradcam, find_target_conv_layer, generate_gradcam_figure
    from src.rsna_dataset import load_dicom_as_array

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] DICOM file not found: {image_path}", file=sys.stderr)
        return 1
    if not image_path.is_file():
        print(f"[ERROR] Specified path is not a file: {image_path}", file=sys.stderr)
        return 1

    if not (0.0 <= args.threshold <= 1.0):
        print(f"[ERROR] Decision threshold must be between 0.0 and 1.0, got {args.threshold}", file=sys.stderr)
        return 1

    model_path = args.model_path if hasattr(args, "model_path") and args.model_path else RSNA_BEST_MODEL_PATH
    print(f"\n[PREDICT-RSNA] Running RSNA DICOM inference on: {image_path}")
    print(f"Target Checkpoint: {model_path} | Decision Threshold: {args.threshold:.2f}")

    try:
        model = load_inference_model(model_path)
        result = predict_rsna_dicom(
            dicom_input=image_path,
            model=model,
            threshold=args.threshold,
            include_patient_id=getattr(args, "include_patient_id", False),
        )

        print("\n" + "=" * 65)
        print("          RSNA DICOM LUNG OPACITY PREDICTION REPORT")
        print("=" * 65)
        print(f"  DICOM File:            {result['dicom_path']}")
        print(f"  Model Checkpoint:      {result['model_checkpoint']}")
        meta = result.get("dicom_metadata", {})
        if meta and "error" not in meta:
            if "patient_id" in meta:
                print(f"  Patient ID (Opt-in):   {meta['patient_id']}")
            print(f"  Modality:              {meta.get('modality', 'CR')}")
            print(f"  View Position:         {meta.get('view_position', 'N/A')}")
            print(f"  Photometric Interp:    {meta.get('photometric_interpretation', 'N/A')}")
            print(f"  Dimensions:            {meta.get('rows', 'N/A')} x {meta.get('columns', 'N/A')}")
        print(f"  Model-Estimated Score: {result['lung_opacity_score']:.4f}")
        print(f"  Decision Threshold:    {result['decision_threshold']:.2f}")
        print(f"  Classification:        {result['predicted_class']}")
        print(f"  Opacity Detected:      {'YES (Positive)' if result['is_opacity_detected'] else 'NO (Negative)'}")
        print(f"  Probabilities:         NO_LUNG_OPACITY: {result['probabilities']['NO_LUNG_OPACITY']:.4f} | LUNG_OPACITY: {result['probabilities']['LUNG_OPACITY']:.4f}")
        print("-" * 65)
        print(f"[DISCLAIMER] {result['disclaimer']}")
        print("=" * 65)

        # Handle optional Grad-CAM
        if getattr(args, "gradcam", False):
            print("\n[EXPLAIN-RSNA] Generating Grad-CAM convolutional saliency map...")
            try:
                img_uint8 = load_dicom_as_array(image_path, target_size=(224, 224))
                tensor = ((img_uint8.astype(np.float32) / 127.5) - 1.0)[np.newaxis, ...]

                target_layer = getattr(args, "target_layer", None) or find_target_conv_layer(model)
                heatmap = generate_gradcam(model, tensor, target_layer_name=target_layer)

                output_plot_path = (
                    Path(args.output)
                    if getattr(args, "output", None)
                    else PLOTS_DIR / f"gradcam_rsna_{image_path.stem}.png"
                )
                output_plot_path.parent.mkdir(parents=True, exist_ok=True)
                if output_plot_path.exists():
                    print(f"[INFO] Existing visualization detected; overwriting: {output_plot_path}")

                generate_gradcam_figure(
                    original_image=img_uint8,
                    heatmap=heatmap,
                    predicted_class=result["predicted_class"],
                    model_score=result["lung_opacity_score"],
                    threshold=result["decision_threshold"],
                    target_layer_name=target_layer,
                    save_path=output_plot_path,
                )
                print(f"[SUCCESS] Multi-panel Grad-CAM visualization saved to: {output_plot_path}")
            except Exception as cam_exc:
                print(f"[WARNING] Grad-CAM generation failed: {type(cam_exc).__name__}: {cam_exc}", file=sys.stderr)

        # Optional JSON output
        if getattr(args, "json_output", None):
            try:
                json_path = Path(args.json_output)
                json_path.parent.mkdir(parents=True, exist_ok=True)
                if json_path.exists():
                    print(f"[INFO] Existing prediction report detected; overwriting: {json_path}")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                print(f"[SUCCESS] Prediction JSON saved to: {json_path}")
            except (PermissionError, OSError) as write_exc:
                print(f"[ERROR] Failed to save prediction JSON to {args.json_output}: {write_exc}", file=sys.stderr)
                return 1

        return 0
    except Exception as exc:
        print(f"\n[ERROR] RSNA inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def run_audit_rsna(args: argparse.Namespace) -> int:
    """Execute the RSNA-specific dataset audit subcommand with structured 5-tier reporting."""
    from src.rsna_dataset import audit_rsna_pipeline

    print(f"\n[AUDIT-RSNA] Auditing RSNA dataset pipeline across 5 verification tiers...")
    report = audit_rsna_pipeline(
        metadata_dir=args.metadata_dir,
        dicom_dir=args.dicom_dir,
        processed_dir=args.processed_dir,
        max_readability_checks=getattr(args, "max_checks", None),
    )

    print("\n" + "=" * 70)
    print("           RSNA DATASET 5-TIER AUDIT REPORT")
    print("=" * 70)

    # Tier 1
    print("\n[Tier 1: Metadata & Annotation Integrity]")
    print(f"  Status:             {report['metadata_status']}")
    print(f"  Total Mappings:     {report.get('mapping_count', 0):,}")
    print(f"  Valid Cohort:       {report.get('total_valid_cohort', 0):,}")
    print(f"  Excluded Cohort:    {report.get('total_excluded_cohort', 0):,}")
    print(f"  Unique Patients:    {report.get('unique_patients_in_valid_cohort', 0):,}")
    class_dist = report.get('class_distribution', {})
    print(f"  Class Distribution: NO_LUNG_OPACITY = {class_dist.get('NO_LUNG_OPACITY', 0):,}, "
          f"LUNG_OPACITY = {class_dist.get('LUNG_OPACITY', 0):,}")

    # Tier 2
    print("\n[Tier 2: Manifest & Patient-Split Integrity]")
    print(f"  Status:             {report['manifest_status']}")
    print(f"  Patient Leakage:    {'ZERO (Strictly Disjoint)' if report.get('patient_leakage_verified_zero') else 'LEAKAGE DETECTED'}")
    ms = report.get("manifest_summary", {}) or {}
    splits = ms.get("splits", {})
    if splits:
        for split_name, s_info in splits.items():
            print(f"    - {split_name.capitalize():<5}: {s_info.get('num_images', 0):,} images ({s_info.get('num_unique_patients', 0):,} patients)")

    # Tier 3
    print("\n[Tier 3: Local DICOM Completeness]")
    print(f"  Status:             {report['completeness_status']}")
    local_count = report.get('local_dicoms_present_count', 0)
    valid_cohort = report.get('total_valid_cohort', 29684)
    pct = (local_count / valid_cohort * 100.0) if valid_cohort else 0.0
    print(f"  Present Locally:    {local_count:,} / {valid_cohort:,} ({pct:.2f}%)")
    print(f"  Missing Locally:    {max(0, valid_cohort - local_count):,}")

    # Tier 4
    print("\n[Tier 4: DICOM Readability Validation]")
    print(f"  Status:             {report['readability_status']}")
    rv = report.get("readability_validation", {})
    print(f"  Checked:            {rv.get('checked_count', 0):,}")
    print(f"  Readable & Decoded: {rv.get('valid_count', 0):,}")
    print(f"  Corrupted / Zero:   {rv.get('corrupt_count', 0):,}")

    # Tier 5
    print("\n[Tier 5: Overall Training-Data Readiness]")
    print(f"  Readiness Status:   {report['overall_readiness']}")
    print(f"  Model Trainable:    {'YES' if report['is_ready_for_training'] else 'NO (Requires complete local dataset)'}")
    print("=" * 70)

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    if report.get("issues"):
        print("\nIssues:")
        for iss in report["issues"]:
            print(f"  [ERROR] {iss}")
        return 1

    return 0


def run_prepare_rsna(args: argparse.Namespace) -> int:
    """Execute patient-level splitting and manifest generation for RSNA cohort."""
    from src.rsna_dataset import (
        build_cohort_records,
        create_patient_level_splits,
        load_rsna_annotations,
        load_rsna_mapping,
        save_manifests,
    )

    meta_dir = Path(args.metadata_dir)
    mapping_p = meta_dir / "pneumonia-challenge-dataset-mappings_2018.json"
    ann_p = meta_dir / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"

    print(f"\n[PREPARE-RSNA] Loading metadata from {meta_dir}...")
    mappings, map_err = load_rsna_mapping(mapping_p)
    annotations, ann_err = load_rsna_annotations(ann_p)

    if map_err:
        print(f"[WARN] Mapping parsing reported {len(map_err)} errors.")

    valid_recs, excluded = build_cohort_records(mappings, annotations, dicom_dir=args.dicom_dir)
    print(f"Classified records: {len(valid_recs)}, Excluded records: {len(excluded)}")

    print(f"Splitting cohort by patient (Seed={args.seed}, Proportions=70/15/15)...")
    splits = create_patient_level_splits(valid_recs, seed=args.seed)

    print(f"Writing manifests to {args.output_dir}...")
    summary = save_manifests(splits, excluded, args.output_dir, dicom_root=args.dicom_dir)
    print("\n[SUCCESS] Manifest generation complete:")
    print(json.dumps(summary, indent=2))
    return 0


def run_download_rsna(args: argparse.Namespace) -> int:
    """Execute download or resume of RSNA adjudicated dataset."""
    from src.downloader import (
        EXPECTED_ARCHIVE_SIZE,
        ResumableDownloader,
        compute_file_sha256,
        format_progress_line,
        safe_extract_archive,
        verify_zip_integrity,
    )

    downloader = ResumableDownloader(
        target_path=args.target_path,
        chunk_size=args.chunk_size_mb * 1024 * 1024,
        max_workers=args.workers,
    )

    print("\n[DOWNLOAD-RSNA] Starting / Resuming RSNA Dataset Transfer...")
    print(f"Target:      {args.target_path}")
    print(f"Workers:     {args.workers}")
    print(f"Chunk Size:  {args.chunk_size_mb} MB")

    prog = downloader.download(
        progress_callback=lambda p: print(format_progress_line(p)),
        max_duration_seconds=args.max_duration,
    )

    print(f"\nFinal Transfer Status: {prog.status}")
    print(f"Downloaded: {prog.downloaded_bytes:,} / {prog.total_bytes:,} bytes ({prog.percent:.2f}%)")

    if prog.status == "COMPLETE":
        print("\n[VERIFICATION] Verifying completed archive size and ZIP integrity...")
        actual_size = Path(args.target_path).stat().st_size
        print(f"Actual Archive Size: {actual_size:,} bytes (Expected: {EXPECTED_ARCHIVE_SIZE:,} bytes)")
        if actual_size != EXPECTED_ARCHIVE_SIZE:
            print("[ERROR] File size mismatch! Refusing to extract corrupt archive.", file=sys.stderr)
            return 1

        print("Testing ZIP CRC-32 integrity of all entries...")
        is_valid, err = verify_zip_integrity(args.target_path)
        if not is_valid:
            print(f"[ERROR] ZIP integrity check failed: {err}", file=sys.stderr)
            return 1
        print("[SUCCESS] ZIP archive passed all integrity and CRC checks!")

        print("Computing SHA-256 checksum (streaming)...")
        sha256 = compute_file_sha256(args.target_path)
        print(f"Archive SHA-256: {sha256}")

        if args.extract:
            print(f"\n[EXTRACTION] Safely extracting DICOM archive to: {args.dicom_dir} ...")
            extracted_cnt, extract_errs = safe_extract_archive(args.target_path, args.dicom_dir)
            print(f"Extracted files: {extracted_cnt:,}")
            if extract_errs:
                print(f"[WARN] Extraction reported {len(extract_errs)} issues.")
            print("[SUCCESS] Extraction completed successfully!")

        return 0

    elif prog.status == "PAUSED":
        print("[PAUSED] Download session paused cleanly. Manifest preserved for resumption.")
        return 0
    else:
        print(f"[ERROR] Download encountered an error: {prog.error_message}", file=sys.stderr)
        return 1


def run_smoke_test_rsna(args: argparse.Namespace) -> int:
    """Execute end-to-end smoke test verifying MobileNetV2 architecture and gradient updates."""
    from src.train import run_rsna_smoke_test

    print("\n[SMOKE-TEST] Executing MobileNetV2 RSNA pipeline smoke test...")
    print(f"Manifest: {args.train_manifest} | DICOMs: {args.dicom_dir} | Batch: {args.batch_size} | Weights: {args.weights}")
    try:
        result = run_rsna_smoke_test(
            train_manifest=args.train_manifest,
            dicom_root=args.dicom_dir,
            weights=None if args.weights.lower() == "none" else args.weights,
            batch_size=args.batch_size,
        )
        print("\n=======================================================")
        print("          RSNA SMOKE TEST: PASSED")
        print("=======================================================")
        print(f"  Input Shape:          {result['input_shape']}")
        print(f"  Output Shape:         {result['output_shape']}")
        print(f"  Batch Tested:         {result['batch_size_tested']} images")
        print(f"  Output Range:         [{result['forward_min']:.4f}, {result['forward_max']:.4f}]")
        print(f"  Loss (1 train step):  {result['loss_after_one_step']:.4f} (Finite)")
        print(f"  Smoke Checkpoint:     {result['smoke_checkpoint_path']}")
        print("=======================================================\n")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Smoke test failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def run_train_rsna(args: argparse.Namespace) -> int:
    """Execute the MobileNetV2 transfer-learning training on RSNA dataset."""
    from src.train import train_rsna_baseline

    print("\n[TRAIN-RSNA] Initiating MobileNetV2 RSNA Transfer Learning Pipeline...")
    print(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | Frozen Backbone: {args.freeze_base}")
    print(f"Train Manifest: {args.train_manifest}")
    print(f"Val Manifest:   {args.val_manifest}")
    print(f"DICOM Root:     {args.dicom_dir}")
    try:
        result = train_rsna_baseline(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            dicom_root=args.dicom_dir,
            model_save_path=args.save_path,
            best_model_path=args.best_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            freeze_base=args.freeze_base,
            weights=None if args.weights.lower() == "none" else args.weights,
            steps_per_epoch=args.steps_per_epoch,
            val_steps=args.val_steps,
            output_dir=args.output_dir,
            plots_dir=args.plots_dir,
        )
        print("\n=======================================================")
        print(f"       RSNA {result['experiment_type']} COMPLETE")
        print("=======================================================")
        print(f"  Experiment Type:      {result['experiment_type']}")
        print(f"  Train Coverage:       {result['train_coverage_percentage']}% ({result['train_images_per_epoch']:,} images/epoch)")
        print(f"  Val Coverage:         {result['val_coverage_percentage']}% ({result['val_images_per_epoch']:,} images/epoch)")
        print(f"  Epochs Completed:     {result['epochs_completed']}")
        print(f"  Duration:             {result['training_duration_seconds']}s")
        print(f"  Final Train Loss:     {result['final_loss']:.4f}")
        print(f"  Final Val Loss:       {result['final_val_loss']:.4f}")
        print(f"  Final Train AUC:      {result['final_auc']:.4f}")
        print(f"  Final Val AUC:        {result['final_val_auc']:.4f}")
        print(f"  Best Checkpoint:      {result['best_model_path']}")
        print(f"  Final Checkpoint:     {result['final_model_path']}")
        print(f"  Metrics History CSV:  {result['history_csv']}")
        print(f"  Convergence Plot:     {result['plot_path']}")
        print("=======================================================\n")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] RSNA Training failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def run_evaluate_rsna(args: argparse.Namespace) -> int:
    """Execute evaluation on the held-out RSNA test split."""
    from src.evaluate import evaluate_rsna_model

    print(f"\n[EVAL-RSNA] Evaluating checkpoint '{args.model_path}' on held-out test split...")
    print(f"Test Manifest: {args.test_manifest} | Threshold: {args.threshold} | Batch: {args.batch_size}")
    try:
        report = evaluate_rsna_model(
            model_path=args.model_path,
            manifest_csv=args.test_manifest,
            dicom_root=args.dicom_dir,
            output_dir=args.output_dir,
            plots_dir=args.plots_dir,
            threshold=args.threshold,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        print("\n=======================================================")
        print("       RSNA HELD-OUT TEST BENCHMARK REPORT")
        print("=======================================================")
        print(f"  Dataset:              {report['dataset_name']}")
        print(f"  Total Test Samples:   {report['total_samples']:,}")
        print(f"  Class Distribution:   NO_OPACITY={report['class_distribution']['NO_LUNG_OPACITY (0)']:,}, OPACITY={report['class_distribution']['LUNG_OPACITY (1)']:,}")
        print(f"  Decision Threshold:   {report['decision_threshold_selected']:.2f}")
        print(f"  ROC-AUC:              {report['auroc']:.4f}" if report['auroc'] is not None else "  ROC-AUC: N/A")
        print(f"  Sensitivity (Recall): {report['sensitivity_recall']:.4f}" if report['sensitivity_recall'] is not None else "  Sensitivity: N/A")
        print(f"  Specificity:          {report['specificity']:.4f}" if report['specificity'] is not None else "  Specificity: N/A")
        print(f"  Precision:            {report['precision']:.4f}" if report['precision'] is not None else "  Precision: N/A")
        print(f"  F1-Score:             {report['f1_score']:.4f}" if report['f1_score'] is not None else "  F1-Score: N/A")
        print(f"  Accuracy:             {report['accuracy']:.4f}" if report['accuracy'] is not None else "  Accuracy: N/A")
        print("  Confusion Matrix:")
        print(f"    [TN={report['true_negatives']}, FP={report['false_positives']}]")
        print(f"    [FN={report['false_negatives']}, TP={report['true_positives']}]")
        if "roc_plot_file" in report:
            print(f"  ROC Plot:             {report['roc_plot_file']}")
        if "confusion_matrix_plot_file" in report:
            print(f"  Confusion Matrix:     {report['confusion_matrix_plot_file']}")
        print(f"  Report File:          {report['report_file']}")
        print(f"\n[DISCLAIMER] {report['clinical_disclaimer']}")
        print("=======================================================\n")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] RSNA Evaluation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        return run_audit(args)
    elif args.command == "audit-rsna":
        return run_audit_rsna(args)
    elif args.command == "download-rsna":
        return run_download_rsna(args)
    elif args.command == "prepare-rsna":
        return run_prepare_rsna(args)
    elif args.command == "smoke-test-rsna":
        return run_smoke_test_rsna(args)
    elif args.command == "train-rsna":
        return run_train_rsna(args)
    elif args.command == "evaluate-rsna":
        return run_evaluate_rsna(args)
    elif args.command == "train":
        return run_train(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    elif args.command == "predict":
        return run_predict(args)
    elif args.command == "predict-rsna":
        return run_predict_rsna(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

