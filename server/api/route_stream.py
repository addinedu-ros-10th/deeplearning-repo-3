from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import cv2, time

router = APIRouter()
last_frame = None   # detect()에서 갱신

def gen_frames():
    global last_frame
    while True:
        if last_frame is None:
            time.sleep(0.05)
            continue
        # YOLO 결과가 그려진 overlay 프레임을 JPEG 인코딩
        _, buffer = cv2.imencode(".jpg", last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)  # 약 20fps 제한

@router.get("/")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=frame")
