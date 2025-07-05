#!/usr/bin/env python3
"""
Water Surface Segmentation Training Script
Train YOLOv11n model for water surface segmentation on beach images.
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11n model for water surface segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml file"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolov11n-seg.pt",
        help="Path to pretrained weights"
    )

    parser.add_argument(
        "--img",
        type=int,
        default=640,
        help="Image size for training"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for training (cpu, cuda, mps)"
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/segment",
        help="Project directory"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="nwsd_train",
        help="Experiment name"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )

    parser.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save model every n epochs"
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input arguments."""
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data configuration file not found: {args.data}")

    if not args.weights.startswith("yolov11") and not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")


def main():
    """Main training function."""
    args = parse_arguments()

    try:
        validate_inputs(args)

        print(f"Loading model: {args.weights}")
        model = YOLO(args.weights)

        train_params = {
            'data': args.data,
            'imgsz': args.img,
            'batch': args.batch,
            'epochs': args.epochs,
            'device': args.device,
            'project': args.project,
            'name': args.name,
            'patience': args.patience,
            'save_period': args.save_period,
            'save': True,
            'verbose': True,
            'plots': True,
            'val': True,
        }

        print("Starting training with parameters:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")

        results = model.train(**train_params)

        model_save_path = os.path.join(args.project, args.name, "weights", "best.pt")
        final_model_path = os.path.join("model", "nwsd-v2.pt")

        os.makedirs("model", exist_ok=True)

        if os.path.exists(model_save_path):
            import shutil
            shutil.copy2(model_save_path, final_model_path)
            print(f"Best model saved to: {final_model_path}")

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
