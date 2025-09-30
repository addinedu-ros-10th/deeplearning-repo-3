import cv2
import requests
import threading
import numpy as np
import pyrealsense2 as rs
import socket, json
from client_api import send_caption, send_vqa  

# HTTP 서버 (YOLO 결과 JSON)
SERVER_URL = "http://192.168.0.155:8000"

# UDP 서버 (프레임 전송)
UDP_IP = "192.168.0.155"   # 서버 IP
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -----------------------------
# 서버 전송 함수 (color + depth) → HTTP
# -----------------------------
def send_frames(color_frame, depth_frame):
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

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
# UDP로 프레임 전송 함수
# -----------------------------
def send_udp_frame(color_frame):
    color_image = np.asanyarray(color_frame.get_data())
    _, encoded = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    sock.sendto(encoded.tobytes(), (UDP_IP, UDP_PORT))

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

            # 화면에 출력 (디버깅용)
            cv2.imshow("RealSense Color", color_image)
            cv2.imshow("RealSense Depth", depth_colormap)

            # -----------------------------
            # YOLO 결과 (HTTP 요청)
            # -----------------------------
            result = send_frames(color_frame, depth_frame)
            if result is None:
                print("Can't send frame")
                continue

            threat = result["threat_level"]
            detections = result["detections"]
            states = result.get("states", {}) 

            if threat == "high":
                print("🚨 위협 감지:", detections, states)
            elif threat == "medium":
                print("⚠️ 주의:", detections, states)
            else:
                print("✅ 안전:", detections, states)

            # -----------------------------
            # 프레임을 UDP로 전송 (실시간 뷰잉 용도)
            # -----------------------------
            send_udp_frame(color_frame)

            # -----------------------------
            # 키 입력 처리 (캡션 / VQA)
            # -----------------------------
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
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
