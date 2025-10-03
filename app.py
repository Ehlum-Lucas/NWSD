
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
import io
import base64
from typing import Optional

app = FastAPI()

MODEL_PATH = "model/nwsd-v2.pt"
model = YOLO(MODEL_PATH)

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image

def postprocess_results(results, original_shape):
    if len(results) == 0 or results[0].masks is None:
        return None, None
    result = results[0]
    masks = result.masks.data.cpu().numpy()  # (N, H, W)
    binary_mask = np.zeros(original_shape, dtype=np.uint8)
    if len(masks) > 0:
        resized_masks = [cv2.resize(mask, (original_shape[1], original_shape[0])) for mask in masks]
        combined_mask = np.max(resized_masks, axis=0)
        binary_mask = (combined_mask > 0.5).astype(np.uint8) * 255
    return binary_mask, masks

def create_overlay(image: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def calculate_water_percentage(binary_mask: np.ndarray) -> float:
    if binary_mask is None:
        return 0.0
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    water_pixels = np.sum(binary_mask > 0)
    return (water_pixels / total_pixels) * 100

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    save_mask: Optional[bool] = False,
    save_overlay: Optional[bool] = False
):
    image_bytes = await file.read()
    image = preprocess_image_bytes(image_bytes)
    original_shape = image.shape[:2]
    # Run inference
    results = model(image, conf=0.25, iou=0.45, verbose=False)
    binary_mask, masks = postprocess_results(results, original_shape)
    overlay = None
    water_percentage = None
    mask_bytes = None
    overlay_bytes = None
    if binary_mask is not None:
        overlay = create_overlay(image, binary_mask)
        water_percentage = calculate_water_percentage(binary_mask)
        if save_mask:
            _, mask_bytes = cv2.imencode('.png', binary_mask)
        if save_overlay:
            _, overlay_bytes = cv2.imencode('.png', overlay)
    else:
        water_percentage = 0.0
    response = {
        "water_percentage": water_percentage,
        "detected": binary_mask is not None
    }
    if save_mask and mask_bytes is not None:
        response["mask_png_base64"] = base64.b64encode(mask_bytes).decode("utf-8")
    if save_overlay and overlay_bytes is not None:
        response["overlay_png_base64"] = base64.b64encode(overlay_bytes).decode("utf-8")
    return JSONResponse(content=response)
