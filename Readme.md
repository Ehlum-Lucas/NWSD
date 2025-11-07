---
language: "en"
license: "gpl-3.0"
tags:
  - segmentation
  - computer-vision
  - yolo
  - beach
  - water
  - open-source
task_categories:
  - image-segmentation
---

# 🌊 Water Surface Segmentation on Beach Images

## Model Overview
This model performs **semantic segmentation of water surfaces** in beach or coastal images.
It’s a fine-tuned version of **YOLOv11n**, adapted for **binary segmentation** with a single class: **`water`**.

Built for lightweight, real-time deployment, the model achieves strong accuracy while remaining small and efficient.

---

## 🧠 Model Details
- **Architecture**: YOLOv11n segmentation head (binary)
- **Base framework**: PyTorch / Ultralytics YOLOv11
- **Input size**: 640×640 RGB images  
- **Output**: Binary segmentation mask (1 class — `water`)
- **Model file**: `nwsd-v2.pt` (≈6 MB)

---

## 🚀 Key Features
- ⚡ **Real-time inference** on CPU/GPU  
- 🖼 **Outputs**: Binary masks, overlays, and coverage statistics  
- 📊 **Evaluation tools** included for metrics & visualization  
- 🐍 **Easy Python integration** via a simple API (`nwsd_api.py`)

---

## 📈 Performance
| Metric | Value | Notes |
|:--|:--|:--|
| **mAP50** | > 0.85 | On validation set |
| **Inference speed** | ~50 ms/image | On CPU |
| **GPU memory** | < 2 GB | For 640×640 input |

---

## 🗂 Dataset
- **Type**: Binary segmentation  
- **Classes**: `water`  
- **Annotations**: PNG masks  
- **Source**: Custom-labeled beach dataset  

🔗 [Dataset on Roboflow](https://universe.roboflow.com/neptune-uxxqf/neptune-water-surface-detection)

---

## 🧩 Intended Uses
**Use cases:**
- Coastal or maritime monitoring  
- Beach safety & drowning prevention systems  
- Environmental analysis (e.g., water coverage estimation)  

**Limitations:**
- Designed for daylight, clear beach imagery  
- May underperform in low-visibility or night-time scenes

---

## 🧪 How to Use

### Load model from Hub
```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(repo_id="Ehlum-Lucas/NWSD", filename="nwsd-v2.pt")
model = torch.load(model_path, map_location="cpu")
model.eval()
```

### Inference example
```python
from PIL import Image
import torch
from torchvision import transforms

img = Image.open("beachTest.jpg").convert("RGB")
input_tensor = transforms.ToTensor()(img).unsqueeze(0)

with torch.no_grad():
    pred = model(input_tensor)
```

### ⚙️ Training
You can fine-tune or retrain the model using YOLOv11 tools:
```bash
python train.py --data data.yaml --weights <path_to_weights> --img 640 --batch 16 --epochs 50
```
Example configuration (data.yaml) defines paths to your datasets and class names.

### 🧭 Evaluation
```bash
python evaluate.py --data data.yaml --weights model/nwsd-v2.pt
```

Generates:

- Binary mask
- Overlay visualization
- Water coverage stats

## License
This model is released under the **GPL-3.0 License**. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this model in your work, please consider citing:
```latex
@misc{nwsd2025,
  title={Water Surface Segmentation on Beach Images},
  author={Lucas Iglesia},
  year={2025},
  howpublished={\url{https://huggingface.co/Ehlum-Lucas/NWSD}}
}
```

