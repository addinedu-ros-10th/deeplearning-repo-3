# sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0 libgl1 libglib2.0-0 mpg123 sounddevice librealsense2-dev
''' 서버 응답
{
  "detections": [
    {"model": "Dynamic", "class": "person", "depth_m": 1.8},
    {"model": "Static", "class": "traffic_light", "depth_m": 4.2}
  ],1
  "states": {
    "ground_left": "warning",
    "ground_right": "safe",
    "head": "safe"
  },
  "threat_level": "high"
}
'''

import cv2
import requests
import threading
import numpy as np
import pyrealsense2 as rs
from client_api import send_caption, send_vqa  

# SERVER_URL = "http://192.168.0.132:8000"
# SERVER_URL = "http://10.26.252.1:8000"
SERVER_URL = "http://192.168.0.155:8000"
# SERVER_URL = "https://uninvigorative-sau-blurredly.ngrok-free.dev"



# -----------------------------
# 서버 전송 함수 (color + depth)
# -----------------------------
def send_frames(color_frame, depth_frame):
    # numpy 변환
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 인코딩 (color는 jpg, depth는 png 보존)
    _, color_encoded = cv2.imencode(".jpg", color_image)
    _, depth_encoded = cv2.imencode(".png", depth_image)

    response = requests.post(
        f"{SERVER_URL}/detect/",
        files={
            "color": ("color.jpg", color_encoded.tobytes(), "image/jpeg"),
            "depth": ("depth.png", depth_encoded.tobytes(), "image/png"),
        }
    )
    if response.status_code != 200:
        print("❌ 서버 오류:", response.status_code, response.text)
        return None
    return response.json()

# -----------------------------
# 메인 루프
# -----------------------------
def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print("no frame")
                continue
            
            # numpy 변환
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 깊이 이미지를 보기 좋게 변환
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 화면에 출력
            cv2.imshow("RealSense Color", color_image)
            cv2.imshow("RealSense Depth", depth_colormap)


            result = send_frames(color_frame, depth_frame)
            if result is None:
                print("Can't send frame")
                continue

            threat = result["threat_level"]
            detections = result["detections"]
            states = result.get("states", {}) 

            # GPIO 동작
            if threat == "high":
                print("🚨 위협 감지:", detections, states)
            elif threat == "medium":
                print("⚠️ 주의:", detections, states)
            else:
                print("✅ 안전:", detections, states)

            # 키 입력 처리 (추가 기능: 캡션 / VQA)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                color_img = np.asanyarray(color_frame.get_data())
                cv2.imwrite("current.jpg", color_img)
                threading.Thread(target=send_caption, args=("current.jpg",), daemon=True).start()

            if key == ord("2"):
                color_img = np.asanyarray(color_frame.get_data())
                cv2.imwrite("current.jpg", color_img)
                threading.Thread(target=send_vqa, args=("current.jpg",), daemon=True).start()

            if key == ord("q"):
                break

    finally:
        pipeline.stop()
        
if __name__ == "__main__":
    main()
