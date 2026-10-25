
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "defect_model.pt"
model = None

def detect_defects(image):
    global model

    if model is None:
        model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=320,
        device="cpu",
        verbose=False
    )

    result = results[0]
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append({
            "defect": result.names[class_id],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()]
        })

    return {
        "count": len(detections),
        "detections": detections,
        "annotated_image": result.plot()
    }
