# src/app.py

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from movable_map import get_movable_status

# Load the trained YOLO model once
model = YOLO("models/yolov8n_movable.pt")


def run_detection(frame: np.ndarray) -> np.ndarray:
    """
    frame: RGB image (H, W, 3) as numpy array (from Gradio: upload or webcam).
    Returns: RGB image with bounding boxes + labels drawn.
    """
    if frame is None:
        return None

    # Gradio gives RGB, YOLO/OpenCV use BGR
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(img_bgr, conf=0.4, verbose=False)
    r = results[0]
    img = r.orig_img  # BGR

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = model.names[cls_id]
        status = get_movable_status(class_name)
        label = f"{class_name} ({status}) {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        y_text = max(h + 6, y1)
        cv2.rectangle(img, (x1, y_text - h - 6), (x1 + w, y_text), (0, 0, 0), -1)
        cv2.putText(img, label, (x1, y_text - 3), font,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Convert BGR back to RGB for Gradio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def main():
    with gr.Blocks(title="Movable / Immovable Object Detection") as demo:
        gr.Markdown("# Movable / Immovable Object Detection (YOLO)")
        gr.Markdown(
            "Upload an image or use your webcam. "
            "Objects will be tagged as **movable** or **immovable**."
        )

        # ---------- IMAGE UPLOAD TAB ----------
        with gr.Tab("Image Upload"):
            img_input = gr.Image(
                label="Upload / Drag & Drop Image",
                type="numpy",
                sources=["upload"],  # only file upload here
            )
            img_output = gr.Image(
                label="Detections",
                type="numpy",
            )
            run_btn = gr.Button("Run Detection")
            run_btn.click(run_detection, inputs=img_input, outputs=img_output)

        # ---------- WEBCAM TAB ----------
        with gr.Tab("Webcam / Live Feed"):
            cam_input = gr.Image(
                label="Webcam",
                type="numpy",
                sources=["webcam"],   # <-- THIS is the correct arg
                streaming=True,       # stream frames continuously
            )
            cam_output = gr.Image(
                label="Live Detections",
                type="numpy",
            )

            # Each new frame from the webcam is sent to run_detection
            cam_input.stream(fn=run_detection, inputs=cam_input, outputs=cam_output)

    demo.launch()


if __name__ == "__main__":
    main()
