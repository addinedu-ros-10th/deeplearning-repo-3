import cv2
import threading
from vqa_package import stt, blip, tts

def image_qa(frame_path):
    text = stt.record_once()
    input, answer = blip.question_image(frame_path, text)
    tts.tts(answer)
    blip.db_insert(input, answer)

def image_caption(frame_path):
    input, answer = blip.caption_image(frame_path)
    tts.tts(answer)
    blip.db_insert(input, answer)

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        results = model(frame, verbose=False)
        ## 이미지 설명
        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            cv2.imwrite("current.jpg", frame)
            threading.Thread(target=image_caption, args=("current.jpg",), daemon=True).start()
        ## 이미지 질문답변
        if key == ord("2"):
            cv2.imwrite("current.jpg", frame)
            threading.Thread(target=image_qa, args=("current.jpg",), daemon=True).start()
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8n Webcam", annotated_frame)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
