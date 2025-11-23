#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on other dataset.

Edit the hyperparameters in the CONFIG section.
"""

from ultralytics import YOLO
from pathlib import Path

# === CONFIG ===
MODEL = "train20"
MODEL_PATH   = f"runs/detect/{MODEL}/weights/best.pt"
DATA_YAML    = "data.yaml"
EPOCHS       = 20
IMGSZ        = 1280
BATCH        = 8
LR0          = 0.002
CLOSE_MOSAIC = 10

# === OUTPUT DIR (auto-generated) ===
OUT_DIR = Path(f"runs/detect/{MODEL}/fine_tune")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === TRAINING ===
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        close_mosaic=CLOSE_MOSAIC,
        project=str(OUT_DIR.parent),  # "runs/detect"
        name=OUT_DIR.name             
    )

