# src/movable_map.py

MOVABLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "sports ball", "skateboard", "surfboard", "tennis racket",
    "skis", "snowboard", "teddy bear"
}

IMMOVABLE_CLASSES = {
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "hair drier", "toothbrush",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake"
}


def get_movable_status(class_name: str) -> str:
    class_name = str(class_name)
    if class_name in MOVABLE_CLASSES:
        return "movable"
    if class_name in IMMOVABLE_CLASSES:
        return "immovable"
    return "unknown"
