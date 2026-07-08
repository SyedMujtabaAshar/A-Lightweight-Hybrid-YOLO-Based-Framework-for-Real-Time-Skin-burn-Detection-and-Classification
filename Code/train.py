"""
============================================================
Experiment 2 (Proposed): YOLOv8n framework + YOLOv11n C3k2 backbone
Lightweight Hybrid YOLO for Skin Burn Detection & Classification
============================================================
Trains the proposed hybrid model with the thesis recipe:
  640x640 | Adam | lr 0.01 | batch 16 | <=300 epochs (patience 30) | 70/20/10

Usage:
  python scripts/train.py
"""

from ultralytics import YOLO


def main():
    # 1. Load the hybrid architecture
    #    (full YOLOv11n C3k2 backbone + YOLOv8n head/neck)
    model = YOLO("models/yolov8n-yolov11n-backbone.yaml")

    # 2. Train with the thesis hyperparameters
    model.train(
        data="dataset/burn_dataset.yaml",
        epochs=300,          # maximum epochs
        patience=30,         # early stopping
        batch=16,
        imgsz=640,
        optimizer="Adam",
        lr0=0.01,
        seed=0,              # reproducibility
        project="runs",
        name="exp2_proposed",
        exist_ok=True,
    )

    # 3. Validate -> per-class Precision / Recall / mAP@0.5 / mAP@0.5:0.95
    metrics = model.val()
    print("\n================ Overall Metrics ================")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")

    # 4. Efficiency stats (params / GFLOPs) -> confirms ~2.25M / 2.25 GFLOPs
    print("\n================ Model Info =====================")
    model.info()


if __name__ == "__main__":
    main()
