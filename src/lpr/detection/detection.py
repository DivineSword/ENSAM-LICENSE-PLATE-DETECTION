from ultralytics import YOLO

_model = None


def _get_model(model_path: str = "models/yolov8n.pt") -> YOLO:
    """Load the model once and reuse it (lazy loading)."""
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model


def detect_objects(frame, model_path: str = "models/yolov8n.pt"):
    model = _get_model(model_path)
    results = model(frame, conf=0.5)
    return results
