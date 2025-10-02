from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from api.blip_captioning import router as caption_router
from api.blip_qa import router as vqa_router
# from api.detect_yolo import router as detect_router
from api.detect_yolo_pipeline import router as detect_router
from api.route_stream import router as stread_router

import threading, socket, cv2, numpy as np, time

app = FastAPI(title="YOLO + BLIP VQA Server")
app.include_router(caption_router, prefix="/caption", tags=["caption"])
app.include_router(vqa_router, prefix="/vqa", tags=["vqa"])
app.include_router(detect_router, prefix="/detect", tags=["detect"])
# app.include_router(stread_router, prefix="/stream", tags=["stream"])

# ========== UDP 프레임 수신 ==========
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
last_udp_frame = None   # 브라우저 스트리밍용

def udp_frame_listener():
    global last_udp_frame
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"✅ UDP 프레임 수신 서버 실행 (포트 {UDP_PORT})")

    while True:
        data, addr = sock.recvfrom(65535)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            last_udp_frame = frame

# UDP 수신 스레드 시작
threading.Thread(target=udp_frame_listener, daemon=True).start()

# ========== FastAPI 스트리밍 엔드포인트 ==========
def gen_udp_frames():
    global last_udp_frame
    while True:
        if last_udp_frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode(".jpg", last_udp_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.05)  # 약 20fps 제한

@app.get("/udpstream")
async def udp_video_feed():
    return StreamingResponse(gen_udp_frames(), media_type="multipart/x-mixed-replace;boundary=frame")