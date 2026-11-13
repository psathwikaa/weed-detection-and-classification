# Cotton Weed Detection and Classification

A Flask web application for cotton field weed classification (15 classes using InceptionV3) and weed detection (12 classes using YOLOv8n).

## What Exists in This Repository

```text
├── app.py                  # Main Flask app (classification, YOLOv8 detection, webcam, analytics)
├── app1.py                 # Standalone Flask app for YOLOv8 image detection (/detect1)
├── check.py                # Basic TensorFlow sanity-check script
├── best1.pt                # Trained YOLOv8n object detection weights
├── my_model_incep.h5       # Trained Keras InceptionV3 classification weights (input: 299x299)
├── static/
│   ├── analysis/           # Saved training curves and validation batch images
│   └── uploads/            # Uploaded test images
└── templates/
    ├── home.html           # Landing page linking to detection, classification, analytics, and Colab
    ├── classification.html # Upload form for 15-class weed classification
    ├── result.html         # Classification result with weed description and prevention tips
    ├── index1.html         # Upload form for YOLOv8 detection (/detect1 JSON output)
    ├── webcam.html         # Trigger page for live webcam detection (/webcamv)
    ├── analytics.html      # Static page displaying training plots and YOLOv8 batch results
    ├── detection.html      # Unused/legacy template
    ├── detectionresult.html# Unused/legacy template
    └── index.html          # Base template from original YOLOv8 Flask starter
```

---

## Models & Recorded Results

### 1. Weed Detection (`best1.pt` — YOLOv8n)
Extracted directly from the `best1.pt` checkpoint metadata:
* **Base Model:** `yolov8n.pt` (Ultralytics `v8.2.2`)
* **Training Config:** `imgsz=416`, `batch=16`, `2` recorded epochs in checkpoint

| Metric | Epoch 1 | Epoch 2 (Saved Checkpoint) |
| :--- | :--- | :--- |
| **mAP@0.5 (`metrics/mAP50(B)`)** | 66.98% (`0.66975`) | **73.82% (`0.73819`)** |
| **mAP@0.5:0.95 (`metrics/mAP50-95(B)`)** | 59.80% (`0.59804`) | **65.65% (`0.65652`)** |
| **Precision (`metrics/precision(B)`)** | 61.92% (`0.61923`) | **67.25% (`0.67250`)** |
| **Recall (`metrics/recall(B)`)** | 62.73% (`0.62730`) | **68.81% (`0.68812`)** |
| **Validation Losses (`box` / `cls` / `dfl`)** | `0.4797` / `1.4114` / `0.8881` | **`0.4718` / `1.1293` / `0.8805`** |

* **12 Classes in `best1.pt` checkpoint:** `Waterhemp`, `MorningGlory`, `Purslane`, `SpottedSpurge`, `Carpetweed`, `Ragweed`, `Eclipta`, `PricklySida`, `PalmerAmaranth`, `Sicklepod`, `Goosegrass`, `Cutleaf`
* **12 Classes mapped in `app.py`:** `CarpetWeeds`, `Crabgrass`, `Eclipta`, `Goosegrass`, `Morningglory`, `Nutsedge`, `PalmerAmaranth`, `Prickly Sida`, `Purslane`, `Ragweed`, `Sicklepod`, `SpottedSpurge`

### 2. Weed Classification (`my_model_incep.h5` — Keras/TensorFlow)
* **Input Size:** `299x299` RGB (preprocessed with `tensorflow.keras.applications.resnet_v2.preprocess_input`)
* **15 Classes (`app.py`):** `CarpetWeeds`, `Crabgrass`, `Eclipta`, `Goosegrass`, `Morningglory`, `Nutsedge`, `PalmerAmaranth`, `Prickly Sida`, `Purslane`, `Ragweed`, `Sicklepod`, `SpottedSpurge`, `SpurredAnoda`, `Swinecress`, `Waterhemp`

Recorded training curves in `static/analysis/`:

| Model | Epochs | Training Accuracy | Validation Accuracy | Plot |
| :--- | :--- | :--- | :--- | :--- |
| **InceptionV3 (`my_model_incep.h5`)** | 50 | ~96.5% – 97.0% | ~72.6% peak (~71.0% final) | `static/analysis/Picture5.png` |
| **Combination (InceptionV3 + ResNet152)** | 30 | ~96.5% | ~76.8% peak (~56.0% final) | `static/analysis/Picture2.png` |

---

## Flask Routes (`app.py`)

| Route | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Renders `home.html` dashboard |
| `/classification.html` | `GET` | Upload page for 15-class weed classification |
| `/classify` | `POST` | Runs `my_model_incep.h5` on uploaded image and redirects to `/result` |
| `/result` | `GET` | Displays predicted weed species, description, and prevention info |
| `/index1.html` | `GET` | Upload page for YOLOv8 detection |
| `/detect1` | `POST` | Runs `best1.pt` on uploaded image and returns JSON (`class`, `confidence`) |
| `/webcam.html` | `GET` | Page to trigger webcam detection |
| `/webcamv` | `POST` | Runs `best1.pt` live on webcam (`source=0, imgsz=640, conf=0.6, show=True`) |
| `/analytics.html` | `GET` | Displays training plots and YOLOv8 validation batch comparisons |

> **Note:** The `/detect` route (`detect_objects` in `app.py`) is an unimplemented placeholder; image detection is served via `/detect1`.

---

## Setup & Running Locally

1. **Install dependencies:**
   ```bash
   pip install flask tensorflow ultralytics torch opencv-python pillow numpy werkzeug
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   Navigate to `http://127.0.0.1:5000`
