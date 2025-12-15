# src/eval.py
from ultralytics import YOLO

def main():
    # Load the trained model
    model = YOLO("models/yolov8n_movable.pt")

    # Evaluate on the same coco128 dataset
    metrics = model.val(
        data="coco128.yaml",
        imgsz=640,
        batch=8,
        plots=True
    )

    print("YOLO metrics:", metrics.results_dict)

if __name__ == "__main__":
    main()
