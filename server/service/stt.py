import whisper
import aiofiles

stt_model = whisper.load_model("base")

async def transcribe(audio_file):
    # 임시 파일 저장
    async with aiofiles.open("temp.wav", "wb") as out:
        content = await audio_file.read()
        await out.write(content)

    # STT 실행
    result = stt_model.transcribe("temp.wav")
    return result["text"]
