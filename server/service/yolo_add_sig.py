from ultralytics import YOLO
import cv2, numpy as np, base64

model = YOLO("yolov8n.pt")

async def detect(image_file):
    # 파일 읽기
    image_bytes = await image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO 추론
    results = model(img, verbose=False)

    # annotated 이미지
    annotated = results[0].plot()
    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise RuntimeError("YOLO 이미지 인코딩 실패")

    annotated_base64 = base64.b64encode(buffer).decode("utf-8")

    # 탐지 결과 리스트
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "class": model.names[cls_id],
            "confidence": conf
        })

    # 위협 수준 간단 로직 (예시)
    threat_level = "high" if any(d["class"] in ["person", "cup"] for d in detections) else "low"

    return annotated_base64, detections, threat_level
