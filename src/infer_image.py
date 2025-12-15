# src/infer_image.py
import cv2
from ultralytics import YOLO
from movable_map import get_movable_status


def draw_label(img, x1, y1, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
    y1 = max(h + 4, y1)
    cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1, y1 - 2), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    # load your trained weights
    model = YOLO("models/yolov8n_movable.pt")

    image_path = "test.jpg"  # change if needed
    results = model(image_path, conf=0.4)  # conf threshold can be tuned

    for r in results:
        img = r.orig_img
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = model.names[cls_id]
            status = get_movable_status(class_name)

            label = f"{class_name} ({status}) {conf:.2f}"
            print(label)  # print in terminal too

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(img, x1, y1, label)

        cv2.imshow("Movable / Immovable Detection - Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
