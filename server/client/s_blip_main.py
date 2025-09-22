# sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0 libgl1 libglib2.0-0 mpg123

import cv2
import threading
# from ultralytics import YOLO

from client_api import send_caption, send_vqa  

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):  # 이미지 캡션
            cv2.imwrite("current.jpg", frame)
            threading.Thread(target=send_caption, args=("current.jpg",), daemon=True).start()

        if key == ord("2"):  # VQA
            cv2.imwrite("current.jpg", frame)
            threading.Thread(target=send_vqa, args=("current.jpg",), daemon=True).start()
        
        # results = model(frame, verbose=False)
        # annotated_frame = results[0].plot()
        # cv2.imshow("YOLO Webcam", annotated_frame)
        cv2.imshow("Webcam", frame)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
