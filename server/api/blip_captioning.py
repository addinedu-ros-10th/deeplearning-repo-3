from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from service import blip, tts

router = APIRouter()

@router.post("/")
async def caption(image: UploadFile = File(...)):
    # BLIP 캡션 생성
    _, caption_text = await blip.caption_image(image)

    # 음성 변환
    print("답변 파일 재생")
    audio_path = await tts.speak(caption_text)

    return FileResponse(audio_path, media_type="audio/mpeg")
print("caption ok")