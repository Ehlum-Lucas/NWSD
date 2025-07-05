#!/usr/bin/env python3
"""
Water Surface Segmentation Inference Script
This script performs inference on beach images to segment water surfaces using YOLOv11n.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform water surface segmentation on beach images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="model/nwsd-v2.pt",
        help="Path to model weights file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: same as input image)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for segmentation"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )

    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save overlay visualization"
    )

    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save binary mask"
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results visualization plot"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (cpu, cuda, mps)"
    )

    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate input arguments."""
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_ext = Path(args.image).suffix.lower()
    if image_ext not in valid_extensions:
        raise ValueError(f"Unsupported image format: {image_ext}")


def load_model(weights_path: str, device: str = "cpu") -> YOLO:
    """Load YOLO model."""
    try:
        model = YOLO(weights_path)
        model.to(device)
        print(f"Model loaded successfully from: {weights_path}")
        print(f"Using device: {device}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def preprocess_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load and preprocess image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_shape = image.shape[:2]  # (height, width)
    return image, original_shape


def postprocess_results(results, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract masks and create binary mask."""
    if len(results) == 0 or results[0].masks is None:
        print("No water surface detected in the image")
        return None, None

    result = results[0]

    masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W)

    binary_mask = np.zeros(original_shape, dtype=np.uint8)

    if len(masks) > 0:
        resized_masks = []
        for mask in masks:
            resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
            resized_masks.append(resized_mask)

        combined_mask = np.max(resized_masks, axis=0)
        binary_mask = (combined_mask > 0.5).astype(np.uint8) * 255

    return binary_mask, masks


def create_overlay(image: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Create overlay visualization."""
    overlay = image.copy()

    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask > 0] = [255, 0, 0]

    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

    return overlay


def save_results(
    image: np.ndarray,
    binary_mask: Optional[np.ndarray],
    overlay: Optional[np.ndarray],
    output_dir: str,
    base_name: str,
    save_mask: bool = False,
    save_overlay: bool = False
) -> None:
    """Save results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    if save_mask and binary_mask is not None:
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, binary_mask)
        print(f"Binary mask saved to: {mask_path}")

    if save_overlay and overlay is not None:
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"Overlay visualization saved to: {overlay_path}")


def display_results(
    image: np.ndarray,
    binary_mask: Optional[np.ndarray],
    overlay: Optional[np.ndarray],
    output_dir: str = ".",
    base_name: str = "result"
) -> None:
    """Display results using matplotlib."""
    num_plots = 1 + (binary_mask is not None) + (overlay is not None)

    plt.figure(figsize=(5 * num_plots, 5))

    plot_idx = 1

    plt.subplot(1, num_plots, plot_idx)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    plot_idx += 1

    if binary_mask is not None:
        plt.subplot(1, num_plots, plot_idx)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Water Surface Mask")
        plt.axis('off')
        plot_idx += 1

    if overlay is not None:
        plt.subplot(1, num_plots, plot_idx)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay Visualization")
        plt.axis('off')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{base_name}_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Results visualization saved to: {plot_path}")
    plt.close()


def calculate_water_percentage(binary_mask: np.ndarray) -> float:
    """Calculate percentage of water surface in the image."""
    if binary_mask is None:
        return 0.0

    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    water_pixels = np.sum(binary_mask > 0)

    return (water_pixels / total_pixels) * 100


def main():
    """Main inference function."""
    args = parse_arguments()

    try:
        validate_inputs(args)

        if args.output is None:
            output_dir = os.path.dirname(args.image)

            if not output_dir:
                output_dir = "."
        else:
            output_dir = args.output

        base_name = Path(args.image).stem

        model = load_model(args.weights, args.device)

        image, original_shape = preprocess_image(args.image)

        print(f"Processing image: {args.image}")
        print(f"Image shape: {image.shape}")

        results = model(
            args.image,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )

        binary_mask, masks = postprocess_results(results, original_shape)

        overlay = None
        if binary_mask is not None:
            overlay = create_overlay(image, binary_mask)

            water_percentage = calculate_water_percentage(binary_mask)
            print(f"Water surface coverage: {water_percentage:.2f}%")

        save_results(
            image,
            binary_mask,
            overlay,
            output_dir,
            base_name,
            save_mask=args.save_mask,
            save_overlay=args.save_overlay
        )

        if args.save_results:
            display_results(image, binary_mask, overlay, output_dir, base_name)

        print("Inference completed successfully!")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
