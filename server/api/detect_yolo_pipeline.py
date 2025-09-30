from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from service import yolo_pipeline

router = APIRouter()

@router.post("/")
async def detect(color: UploadFile = File(...), depth: UploadFile = File(...)):
    detections, states, threat_level = await yolo_pipeline.detect(color, depth)
    return JSONResponse(content={
        "detections": detections,   # 객체별 class, depth, 위치 등
        "states": states,           # ground_left / ground_right / head
        "threat_level": threat_level
    })