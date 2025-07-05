#!/usr/bin/env python3
"""
NWSD API - Simple Python API for water surface detection
This module provides a simple interface for water surface segmentation.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Union
from pathlib import Path
from ultralytics import YOLO


class WaterSurfaceDetector:
    """Water Surface Detection API using YOLOv11n."""

    def __init__(self, weights_path: str = "model/nwsd-v2.pt", device: str = "cpu"):
        """
        Initialize the water surface detector.

        Args:
            weights_path: Path to model weights
            device: Device to use for inference (cpu, cuda, mps)
        """
        self.weights_path = weights_path
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        self.model = YOLO(self.weights_path)
        self.model.to(self.device)

    def detect(self,
               image: Union[str, np.ndarray],
               conf: float = 0.25,
               iou: float = 0.45) -> Dict:
        """
        Detect water surfaces in an image.

        Args:
            image: Path to image file or numpy array
            conf: Confidence threshold
            iou: IoU threshold for NMS

        Returns:
            Dictionary containing detection results
        """
        if isinstance(image, str):
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not load image: {image}")
            image_path = image
        else:
            img_array = image
            image_path = None

        results = self.model(image_path if image_path else img_array,
                           conf=conf, iou=iou, verbose=False)

        return self._process_results(results, img_array)

    def _process_results(self, results, original_image: np.ndarray) -> Dict:
        """Process YOLO results into structured output."""
        h, w = original_image.shape[:2]

        output = {
            "detected": False,
            "binary_mask": None,
            "overlay": None,
            "water_percentage": 0.0,
            "water_pixels": 0,
            "total_pixels": h * w,
            "bounding_boxes": [],
            "confidence_scores": []
        }

        if len(results) == 0 or results[0].masks is None:
            return output

        result = results[0]

        masks = result.masks.data.cpu().numpy()

        if len(masks) == 0:
            return output

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for mask in masks:
            resized_mask = cv2.resize(mask, (w, h))
            combined_mask = np.maximum(combined_mask, (resized_mask > 0.5).astype(np.uint8))

        binary_mask = combined_mask * 255

        overlay = original_image.copy()
        colored_mask = np.zeros_like(original_image)
        colored_mask[binary_mask > 0] = [0, 0, 255]
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

        water_pixels = np.sum(binary_mask > 0)
        water_percentage = (water_pixels / (h * w)) * 100

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            output["bounding_boxes"] = boxes.tolist()
            output["confidence_scores"] = scores.tolist()

        output.update({
            "detected": True,
            "binary_mask": binary_mask,
            "overlay": overlay,
            "water_percentage": water_percentage,
            "water_pixels": int(water_pixels)
        })

        return output

    def detect_batch(self,
                     image_paths: list,
                     conf: float = 0.25,
                     iou: float = 0.45) -> Dict:
        """
        Detect water surfaces in multiple images.

        Args:
            image_paths: List of paths to image files
            conf: Confidence threshold
            iou: IoU threshold for NMS

        Returns:
            Dictionary with results for each image
        """
        results = {}

        for image_path in image_paths:
            try:
                result = self.detect(image_path, conf, iou)
                results[image_path] = result
            except Exception as e:
                results[image_path] = {"error": str(e)}

        return results

    def save_results(self,
                     results: Dict,
                     output_dir: str,
                     base_name: str,
                     save_mask: bool = True,
                     save_overlay: bool = True) -> Dict[str, str]:
        """
        Save detection results to files.

        Args:
            results: Results from detect() method
            output_dir: Directory to save results
            base_name: Base name for output files
            save_mask: Whether to save binary mask
            save_overlay: Whether to save overlay

        Returns:
            Dictionary with saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        if save_mask and results["binary_mask"] is not None:
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, results["binary_mask"])
            saved_files["mask"] = mask_path

        if save_overlay and results["overlay"] is not None:
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, results["overlay"])
            saved_files["overlay"] = overlay_path

        return saved_files

    def get_water_classification(self, percentage: float) -> str:
        """Classify water coverage level."""
        if percentage < 10:
            return "minimal"
        elif percentage < 30:
            return "low"
        elif percentage < 50:
            return "moderate"
        elif percentage < 70:
            return "high"
        else:
            return "very_high"


# Example usage
def main():
    """Example usage of the WaterSurfaceDetector API."""
    print("🌊 NWSD API Example")
    print("=" * 30)

    detector = WaterSurfaceDetector()

    # Look for test images
    test_images = list(Path("..").glob("*.jpg"))

    if not test_images:
        print("No test images found")
        return

    test_image = str(test_images[0])
    print(f"Processing: {test_image}")

    results = detector.detect(test_image)

    print(f"Water detected: {results['detected']}")
    print(f"Water coverage: {results['water_percentage']:.2f}%")
    print(f"Classification: {detector.get_water_classification(results['water_percentage'])}")

    # Save results
    if results['detected']:
        output_dir = "api_results"
        base_name = Path(test_image).stem
        saved_files = detector.save_results(results, output_dir, base_name)
        print(f"Results saved to: {saved_files}")


if __name__ == "__main__":
    main()
