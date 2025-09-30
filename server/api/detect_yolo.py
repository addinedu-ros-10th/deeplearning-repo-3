from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
from service import yolo

router = APIRouter()

@router.post("/")
async def detect(image: UploadFile = File(...)):
    annotated_bytes = await yolo.detect(image)
    return Response(content=annotated_bytes, media_type="image/jpeg")
print("yolo ok")