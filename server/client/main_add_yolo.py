# sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0 libgl1 libglib2.0-0 mpg123

import cv2
import requests
import threading
import numpy as np
from client_api import send_caption, send_vqa  

SERVER_URL = "http://192.168.0.132:8000"

def send_frame(frame):
    # JPEG 인코딩 후 서버 전송
    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post(
        f"{SERVER_URL}/detect/",   # 여기서 detect 엔드포인트 붙임
        files={"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
    )
    return response.content  # annotated JPEG

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # annotated 이미지 받아오기
        annotated_bytes = send_frame(frame)
        nparr = np.frombuffer(annotated_bytes, np.uint8)
        annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if annotated is None:
            print("❌ 서버에서 받은 이미지 디코딩 실패")
            continue


        # 화면 출력
        cv2.imshow("YOLO Detection (Server)", annotated)

        key = cv2.waitKey(1) & 0xFFlgm-364@cohesive-keel-463404-g9.iam.gserviceaccount.com
        if key == ord("1"):  # 이미지 캡션
            cv2.imwrite("current.jpg", frame)
            threading.Thread(target=send_caption, args=("current.jpg",), daemon=True).start()

        if key == ord("2"):  # VQA
            cv2.imwrite("current.jpg", frame)
            threading.Thread(target=send_vqa, args=("current.jpg",), daemon=True).start()

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
