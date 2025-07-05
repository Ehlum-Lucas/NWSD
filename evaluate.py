#!/usr/bin/env python3
"""
Water Surface Segmentation Evaluation Script
Evaluate the trained model on a validation dataset.
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate water surface segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to validation dataset or data.yaml file"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="model/nwsd-v2.pt",
        help="Path to model weights file"
    )

    parser.add_argument(
        "--img",
        type=int,
        default=640,
        help="Image size for evaluation"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for evaluation (cpu, cuda, mps)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/segment",
        help="Project directory for results"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="nwsd_eval",
        help="Experiment name"
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results in JSON format"
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save results in TXT format"
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate evaluation plots"
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input arguments."""
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data path not found: {args.data}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")


def main():
    """Main evaluation function."""
    args = parse_arguments()

    try:
        validate_inputs(args)

        print(f"Loading model: {args.weights}")
        model = YOLO(args.weights)

        eval_params = {
            'data': args.data,
            'imgsz': args.img,
            'batch': args.batch,
            'conf': args.conf,
            'iou': args.iou,
            'device': args.device,
            'project': args.project,
            'name': args.name,
            'save_json': args.save_json,
            'save_txt': args.save_txt,
            'plots': args.plots,
            'verbose': True,
        }

        print("Starting evaluation with parameters:")
        for key, value in eval_params.items():
            print(f"  {key}: {value}")

        results = model.val(**eval_params)

        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)

        if hasattr(results, 'box') and results.box is not None:
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")

        if hasattr(results, 'seg') and results.seg is not None:
            print(f"Segmentation mAP50: {results.seg.map50:.4f}")
            print(f"Segmentation mAP50-95: {results.seg.map:.4f}")

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
