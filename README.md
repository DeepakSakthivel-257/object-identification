Movable vs Immovable Object Detection (YOLOv8)

OVERVIEW
This project implements a real-time object detection system that identifies objects from images or live video and classifies them as movable or immovable. A simple GUI allows users to upload images or use a webcam for detection.

ALGORIYHM USED
The system uses YOLOv8, a CNN-based single-stage object detection algorithm. YOLO detects object classes and bounding boxes in one forward pass, enabling real-time performance. A rule-based layer is applied on top of YOLO’s output to categorize objects as movable or immovable.

FEATURES
-Image and webcam-based object detection
-Object name and confidence score display
-Movable / immovable classification
-Simple GUI using Gradio
-Runs in real time on CPU

ENVIRONMENT & ML FRAMEWORK USED
-Python Environment: Virtual environment (yolo-env) to isolate dependencies
-Machine Learning Framework: PyTorch (backend used by YOLOv8)
-Object Detection Model: YOLOv8 (Ultralytics implementation)
-GUI Framework: Gradio (for image upload and live webcam interface)
-Supporting Libraries: OpenCV, NumPy

PROJECT STRUCTURE

objects/
├── models/yolov8n_movable.pt
├── src/
│   ├── app.py
│   ├── train.py
│   ├── eval.py
│   └── movable_map.py
├── runs/
└── README.md

DATASET
-COCO128 (subset of COCO dataset)
-Automatically downloaded during training
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset?utm_source

HOW TO RUN
pip install ultralytics gradio opencv-python numpy
python src/app.py

OUTCOME 
The model accurately detects objects and determines whether they are movable or immovable in real time, demonstrating practical scene understanding using deep learning.
