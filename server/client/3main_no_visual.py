# sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0 libgl1 libglib2.0-0 mpg123 sounddevice librealsense2-dev
''' 서버 응답
{
  "detections": [
    {"model": "Dynamic", "class": "person", "depth_m": 1.8},
    {"model": "Static", "class": "traffic_light", "depth_m": 4.2}
  ],
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
import Jetson.GPIO as GPIO
import pyrealsense2 as rs
from client_api import send_caption, send_vqa  

SERVER_URL = "http://192.168.0.132:8000"

# -----------------------------
# GPIO 핀 정의
# -----------------------------
MOTOR1_PIN = 32
MOTOR2_PIN = 33
BTN1_PIN = 16
BTN2_PIN = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(MOTOR1_PIN, GPIO.OUT)
GPIO.setup(MOTOR2_PIN, GPIO.OUT)
GPIO.setup(BTN1_PIN, GPIO.IN)
GPIO.setup(BTN2_PIN, GPIO.IN)

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

            result = send_frames(color_frame, depth_frame)
            if result is None:
                print("Can't send frame")
                continue

            threat = result["threat_level"]
            detections = result["detections"]
            states = result.get("states", {}) 

            # GPIO 동작
            if threat == "high":
                GPIO.output(MOTOR1_PIN, GPIO.HIGH)
                GPIO.output(MOTOR2_PIN, GPIO.HIGH)
                print("⚠️ 위협 감지:", detections, states)
            elif threat == "medium":
                GPIO.output(MOTOR1_PIN, GPIO.HIGH)
                GPIO.output(MOTOR2_PIN, GPIO.LOW)
                print("⚠️ 주의:", detections, states)
            else:
                GPIO.output(MOTOR1_PIN, GPIO.LOW)
                GPIO.output(MOTOR2_PIN, GPIO.LOW)
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
        GPIO.cleanup()

if __name__ == "__main__":
    main()
