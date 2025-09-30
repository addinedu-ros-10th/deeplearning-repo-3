from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from service import stt, blip, tts

router = APIRouter()

@router.post("/")
async def vqa(audio: UploadFile = File(...), image: UploadFile = File(...)):
    # 음성을 질문 텍스트로 변환
    question = await stt.transcribe(audio)

    # BLIP VQA 실행
    _, answer = await blip.question_image(image, question)
    # 음성 변환
    print("답변 파일 재생")
    audio_path = await tts.speak(answer)

    return FileResponse(audio_path, media_type="audio/mpeg")
print("qa ok")