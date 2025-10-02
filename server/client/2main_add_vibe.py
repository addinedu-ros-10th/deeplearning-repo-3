# sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0 libgl1 libglib2.0-0 mpg123 sounddevice

import cv2
import requests
import threading
import numpy as np
import base64
import Jetson.GPIO as GPIO
from client_api import send_caption, send_vqa  

SERVER_URL = "http://192.168.0.132:8000"

# GPIO 핀 정의
MOTOR_PIN = 32
GPIO.setmode(GPIO.BOARD)   # 또는 GPIO.BCM (환경에 맞게)
GPIO.setup(MOTOR_PIN, GPIO.OUT)
pwm = GPIO.PWM(MOTOR_PIN, 200)  # 200Hz
pwm.start(0)

def send_frame(frame):
    # JPEG 인코딩 후 서버 전송
    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(
        f"{SERVER_URL}/detect/",
        files={"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
    )
    if response.status_code != 200:
        print("❌ 서버 오류:", response.status_code, response.text)
        return None
    return response.json()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 서버 응답 받기 (JSON)
            result = send_frame(frame)
            if result is None:
                continue

            # annotated 이미지 복원
            annotated_bytes = base64.b64decode(result["annotated"])
            nparr = np.frombuffer(annotated_bytes, np.uint8)
            annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if annotated is None:
                print("❌ 서버에서 받은 이미지 디코딩 실패")
                continue

            # 화면 출력
            cv2.imshow("YOLO Detection (Server)", annotated)

            # 위협 수준에 따라 모터 제어
            if result["threat_level"] == "high":
                pwm.ChangeDutyCycle(80)  # 강하게 진동
            else:
                pwm.ChangeDutyCycle(0)   # 멈춤

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):  # 이미지 캡션
                cv2.imwrite("current.jpg", frame)
                threading.Thread(target=send_caption, args=("current.jpg",), daemon=True).start()

            if key == ord("2"):  # VQA
                cv2.imwrite("current.jpg", frame)
                threading.Thread(target=send_vqa, args=("current.jpg",), daemon=True).start()

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pwm.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
