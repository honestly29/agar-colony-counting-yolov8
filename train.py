from ultralytics import YOLO
import albumentations as A
import utils.config as cfg


def main():
    model = YOLO(cfg.MODEL)

    model.train(
        data=cfg.DATA,
        epochs=cfg.EPOCHS,
        imgsz=cfg.IMGSZ,
        patience=cfg.PATIENCE,
        batch=cfg.BATCH,
        workers=cfg.WORKERS,
        close_mosaic=cfg.CLOSE_MOSAIC,
    )

if __name__ == "__main__":
    main()

