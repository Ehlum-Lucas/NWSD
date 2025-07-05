# 🌊 Water Surface Segmentation on Beach Images

This project provides an open-source solution for **segmenting water surfaces in beach images** using deep learning. The model is a fine-tuned version of **YOLOv11n**, adapted for binary segmentation using a custom-labeled dataset with a single class: **"water"**.

## 📦 Features

- ⚡ **Fast inference**: Optimized for real-time detection
- 🧠 **YOLOv11n backbone**: Lightweight, efficient, and accurate
- 🖼 **Multiple outputs**: Binary masks, overlays, and water coverage stats
- 🔧 **Flexible deployment**: Works on CPU, GPU, and exportable to ONNX
- 📊 **Comprehensive evaluation**: Built-in performance metrics and visual diagnostics
- 🐍 **Easy integration**: Simple Python API (`nwsd_api.py`) for custom use cases

---

## 📈 Model Performance

- **mAP50**: >0.85 on validation set
- **Inference speed**: ~50ms per image on CPU
- **Memory usage**: <2GB on GPU
- **Model file**: `nwsd-v2.pt` (6.07 MB)

---

## 🗂 Dataset

- **Type**: Binary segmentation (1 class: `water`)
- **Annotation format**: PNG masks
- **Source**: Custom-labeled beach images

🔗 [Download Dataset on Roboflow](https://universe.roboflow.com/neptune-uxxqf/neptune-water-surface-detection)

---

## ⚙️ Installation

```bash
git clone https://github.com/Ehlum-Lucas/NWSD.git
cd NWSD
pip install -r requirements.txt
```

## Quick Start

### Inference

```bash
python3 predict.py --image beachTest.jpg --weights yolo11n-seg.pt
```

**Optionnal flags:**
- `--save-mask`: Save binary mask as `.png`
- `--save-overlay`: Save overlay image
- `--save-results`: Save final visualization plot
- `--conf 0.5`: Set confidence threshold (default: 0.5)
- `--device cuda`: Use GPU for inference (default: CPU)

### Training

```bash
python3 train.py --data data.yaml --weights yolo11n-seg.pt --img 640 --batch 16 --epochs 50
```

An example configuration file `data.yaml` is provided, which specifies paths to training and validation datasets, as well as class names.

### Evaluation

```bash
python3 evaluate.py --data data.yaml --weights model/nwsd-v2.pt
```

## Output Description
The model generates:
- **Binary Mask**: A black-and-white image where white pixels represent water surfaces.
- **Overlay Visualization**: An image showing the original with the water mask overlaid in red
- **Stats Plot**: A plot visualizing coverage, confidence, and other metrics.

## Model Architecture

- **Base Model**: YOLOv11n (nano)
- **Head**: Custom segmentation decoder
- **Input**: RGB image, 640x640
- **Output**: Binary segmentation map (1 class: water)
- **Format**: PyTorch `.pt` model

Contributions are welcome!