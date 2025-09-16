import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# ------------------------------
# 1. YOLO 모델 불러오기
# ------------------------------
# 경량 모델 추천 (속도/VRAM 고려)
model = YOLO("yolov8n.pt")  # yolov8n/s/m 가능

# ------------------------------
# 2. RealSense 파이프라인 설정
# ------------------------------
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)



# Align 객체 생성 (Depth와 Color 정합)
align_to = rs.stream.color
align = rs.align(align_to)

# 파이프라인 시작
pipeline.start(config)

try:
    while True:
        # ------------------------------
        # 3. 프레임 받기
        # ------------------------------
        frames = pipeline.wait_for_frames(timeout_ms=10000)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # ------------------------------
        # 4. YOLO 추론
        # ------------------------------
        results = model(color_image, conf=0.5)

        # 결과를 OpenCV에 그리기
        annotated_frame = results[0].plot()
        
        # ------------------------------
        # 5. 탐지된 객체의 중심에서 깊이 추출
        # ------------------------------
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            cls = int(box.cls[0])
            label = model.names[cls]

            # 중심 좌표 계산
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 깊이값 추출 (mm 단위)
            depth = depth_frame.get_distance(cx, cy)

            # 화면에 표시
            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated_frame,
                f"{label}: {depth:.2f}m",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # ------------------------------
        # 6. 출력
        # ------------------------------
        cv2.imshow("YOLO + Depth", annotated_frame)

        # ESC 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
