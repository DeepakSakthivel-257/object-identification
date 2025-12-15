# src/infer_webcam.py
import cv2
from ultralytics import YOLO
from movable_map import get_movable_status

def main():
    model = YOLO("models/yolov8n_movable.pt")  # or yolov8n.pt if not fine-tuned
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            img = r.orig_img
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = model.names[cls_id]
                status = get_movable_status(class_name)

                label = f"{class_name} ({status}) {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, label, (x1, max(0, y1 - 5)), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, label, (x1, max(0, y1 - 5)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Movable / Immovable Detection", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
