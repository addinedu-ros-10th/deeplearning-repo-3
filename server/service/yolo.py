from ultralytics import YOLO
import cv2
import numpy as np
import io
from api import route_stream

model = YOLO("yolov8n.pt")  # 서버 GPU에서 로드

async def detect(image_file):
    # 파일 읽기
    image_bytes = await image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO 추론
    results = model(img, verbose=False)

    # 시각화된 프레임 얻기
    annotated = results[0].plot()

    # JPEG 인코딩
    success, buffer = cv2.imencode(".jpg", annotated)
    route_stream.last_frame = buffer.copy()
    if not success:
        raise RuntimeError("YOLO 이미지 인코딩 실패")

    return buffer.tobytes()