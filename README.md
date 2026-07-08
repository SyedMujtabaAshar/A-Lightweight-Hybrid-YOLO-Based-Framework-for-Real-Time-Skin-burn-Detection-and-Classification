# A Lightweight Hybrid YOLO Framework for Real-Time Skin Burn Detection and Classification

A lightweight object-detection model that localizes skin burns and classifies their
severity into **first-, second-, and third-degree** in real time.

The proposed model (Experiment 2) keeps the **YOLOv8n** detection framework but replaces
its backbone with the lighter, **C3k2-based YOLOv11n backbone** — achieving competitive
accuracy at a fraction of the computational cost.

---

## Highlights

| Metric | Proposed Model |
|---|---|
| Accuracy | **81.7%** |
| Parameters | **2.25 M** |
| GFLOPs | **2.25** |
| Model size | **4.60 MB** |
| Speed | **11.69 FPS** |
| mAP@0.5 | 81.8% |
| mAP@0.5:0.95 | 81.7% |

### Per-class results

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| First-degree | 74.3% | 83.8% | 82.8% | 82.6% |
| Second-degree | 59.4% | 79.6% | 75.9% | 75.7% |
| Third-degree | 86.7% | 73.2% | 86.8% | 86.7% |
| **Overall** | **73.5%** | **78.9%** | **81.8%** | **81.7%** |

---

## Repository structure

```
burn-detection-yolo/
├── dataset/
│   ├── burn_dataset.yaml         # dataset config (paths + classes)
│   ├── images/{train,val,test}/  # <-- add your images here
│   └── labels/{train,val,test}/  # <-- YOLO-format labels
├── models/
│   └── yolov8n-yolov11n-backbone.yaml   # proposed hybrid architecture
├── scripts/
│   ├── train.py                  # train Experiment 2
│   ├── evaluate.py               # per-class metrics
│   └── predict.py                # inference + per-class confidence
├── runs/
│   └── exp2_proposed/weights/best.pt   # <-- add your trained weights here
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/burn-detection-yolo.git
cd burn-detection-yolo
pip install -r requirements.txt
```

## Training (Experiment 2)

```bash
python scripts/train.py
```

Training recipe (matches the thesis):
`640×640 · Adam · lr 0.01 · batch 16 · ≤300 epochs (patience 30) · 70/20/10 split`

## Evaluation

```bash
python scripts/evaluate.py --weights runs/exp2_proposed/weights/best.pt
```

## Inference + per-class confidence

```bash
python scripts/predict.py \
    --weights runs/exp2_proposed/weights/best.pt \
    --source dataset/images/test
```

---

## Model architecture

The proposed hybrid keeps YOLOv8n's anchor-free detection head and PAN-FPN neck, and
replaces the backbone with the full **YOLOv11n backbone built on C3k2 blocks**
(a lighter, more efficient successor to YOLOv8's C2f block). See
[`models/yolov8n-yolov11n-backbone.yaml`](models/yolov8n-yolov11n-backbone.yaml).

```
Input → [YOLOv11n C3k2 backbone] → [PAN-FPN neck] → [YOLOv8n head] → 3 severity classes
```

---

## Notes

- The dataset is annotated with bounding boxes (CVAT) and validated by a dermatologist.
- Augmentation: rotation and brightness adjustment.
- Confidence scores reported per class are the mean of the model's per-detection
  confidence (`box.conf`), which is distinct from precision/recall/mAP.

## Citation

If you use this work, please cite the associated MS thesis (FAST-NUCES).

## License

Released under the MIT License. YOLO components are subject to the
Ultralytics AGPL-3.0 license.
