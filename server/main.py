from fastapi import FastAPI
from api.blip_captioning import router as caption_router
from api.blip_qa import router as vqa_router
# from api.detect_yolo import router as detect_router
from api.detect_yolo_pipeline import router as detect_router
from api.route_stream import router as stread_router

app = FastAPI(title="YOLO + BLIP VQA Server")
app.include_router(caption_router, prefix="/caption", tags=["caption"])
app.include_router(vqa_router, prefix="/vqa", tags=["vqa"])
app.include_router(detect_router, prefix="/detect", tags=["detect"])
app.include_router(stread_router, prefix="/stream", tags=["stream"])

import pymysql, socket, time, threading


# -------------------------------
# heartbeat 함수
# -------------------------------
def heartbeat_loop():
    conn = pymysql.connect(
        host="database-1.ct0kcwawch43.ap-northeast-2.rds.amazonaws.com",
        user="robot", password="0310", db="bhc_database", autocommit=True
    )
    cur = conn.cursor()
    COMPONENT = "BHC_SERVER"
    IP = "192.168.0.132"

    while True:
        try:
            cur.execute("""
                INSERT INTO system_heartbeat (component, ip, status, last_seen)
                VALUES (%s, %s, 'OK', NOW())
                ON DUPLICATE KEY UPDATE
                  status=VALUES(status),
                  last_seen=NOW()
            """, (COMPONENT, IP))
        except Exception as e:
            print("❌ heartbeat 에러:", e)
        time.sleep(5)

# -------------------------------
# FastAPI 시작 시 heartbeat 실행
# -------------------------------
@app.on_event("startup")
def start_heartbeat():
    t = threading.Thread(target=heartbeat_loop, daemon=True)
    t.start()