# src/train.py
from ultralytics import YOLO

def main():
    # Start from pretrained YOLOv8n on COCO
    model = YOLO("yolov8n.pt")

    # Train (fine-tune) on coco128
    model.train(
        data="coco128.yaml",          # small built-in dataset
        epochs=5,                     # 5 is enough on CPU; you already did 20 once
        imgsz=640,
        batch=8,
        name="movable_immovable_coco4"  # run name
    )

    # DO NOT reload weights or save anything else here.
    # Best weights are automatically saved to:
    # runs/detect/movable_immovable_coco4/weights/best.pt

if __name__ == "__main__":
    main()
